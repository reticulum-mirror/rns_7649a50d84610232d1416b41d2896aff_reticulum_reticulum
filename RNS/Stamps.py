# Reticulum License
#
# Copyright (c) 2016-2026 Mark Qvist
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# - The Software shall not be used in any kind of system which includes amongst
#   its functions the ability to purposefully do harm to human beings.
#
# - The Software shall not be used, directly or indirectly, in the creation of
#   an artificial intelligence, machine learning or language model training
#   dataset, including but not limited to any use that contributes to the
#   training or development of such a model or algorithm.
#
# - The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import RNS
import RNS.vendor.umsgpack as msgpack

import os
import sys
import time
import math
import itertools
import contextlib
import multiprocessing

WORKBLOCK_EXPAND_ROUNDS     = 3000
STAMP_SIZE                  = 32
PN_VALIDATION_POOL_MIN_SIZE = 256
USE_WORKER_MANAGER          = False
POOLING_THRESHOLD_COST      = 15
STOP_CHECK_INTERVAL         = 64

active_jobs = {}

if sys.version_info[0] >= 3 and sys.version_info[1] >= 14:
    USE_WORKER_MANAGER = True

@contextlib.contextmanager
def worker_context(ctx, stamp_cost, workblock, message_id):
    stop_event = ctx.Event()
    result_queue = ctx.Queue(1)
    rounds_queue = ctx.Queue()

    active_jobs[message_id] = [stop_event, result_queue]
    job_procs = []

    try: yield stop_event, result_queue, rounds_queue, job_procs
    finally:
        if message_id in active_jobs: del active_jobs[message_id]
        stop_event.set()

        for p in job_procs:
            if p.is_alive(): p.terminate()
        
        for p in job_procs: p.join(timeout=0.5)
        
        result_queue.close()
        result_queue.join_thread()
        rounds_queue.close()
        rounds_queue.join_thread()
        stop_event.clear()

def stamp_workblock(material, expand_rounds=WORKBLOCK_EXPAND_ROUNDS, no_expand=False):
    if no_expand or not expand_rounds: return material
    wb_st = time.time()
    workblock = b""
    for n in range(expand_rounds):
        workblock += RNS.Cryptography.hkdf(length=256, derive_from=material,
                                           salt=RNS.Identity.full_hash(material+msgpack.packb(n)),
                                           context=None)
    wb_time = time.time() - wb_st

    return workblock

def stamp_value(workblock, stamp):
    material = RNS.Identity.full_hash(workblock+stamp)
    return 256 - int.from_bytes(material, byteorder="big").bit_length()

def stamp_valid(stamp, target_cost, workblock):
    target = 0b1 << 256-target_cost
    result = RNS.Identity.full_hash(workblock+stamp)
    if int.from_bytes(result, byteorder="big") > target: return False
    else: return True

def generate_stamp(message_id, stamp_cost, expand_rounds=WORKBLOCK_EXPAND_ROUNDS, no_expand=False, size=STAMP_SIZE):
    if stamp_cost < 0 or stamp_cost > RNS.Transport.STAMP_COST_CAP: raise ValueError("Invalid stamp cost, cannot generate")
    RNS.log(f"Generating stamp with cost {stamp_cost} for <{RNS.hexrep(message_id[:16], delimit=False)}{'…' if len(message_id) > 16 else ''}>", RNS.LOG_DEBUG)
    workblock = stamp_workblock(message_id, expand_rounds=expand_rounds, no_expand=no_expand)
    
    start_time = time.time()
    stamp = None
    rounds = 0
    value = 0

    if RNS.vendor.platformutils.is_windows() or RNS.vendor.platformutils.is_darwin():
        RNS.log(f"Multi-core stamp generation on this platform is not supported, work limited to single CPU core.", RNS.LOG_DEBUG) if RNS.sl(RNS.LOG_DEBUG) else None

    if RNS.vendor.platformutils.is_windows() or RNS.vendor.platformutils.is_darwin(): stamp, rounds = job_simple(stamp_cost, workblock, message_id, size)
    elif RNS.vendor.platformutils.is_android(): stamp, rounds = job_android(stamp_cost, workblock, message_id, size)
    else:
        if USE_WORKER_MANAGER: stamp, rounds = job_linux_managed(stamp_cost, workblock, message_id, size) if stamp_cost >= POOLING_THRESHOLD_COST else job_simple(stamp_cost, workblock, message_id, size)
        else:                  stamp, rounds = job_linux(stamp_cost, workblock, message_id, size) if stamp_cost >= POOLING_THRESHOLD_COST else job_simple(stamp_cost, workblock, message_id, size)
    
    duration = max(time.time()-start_time, 1E-6)
    speed = rounds/duration
    if stamp != None: value = stamp_value(workblock, stamp)

    RNS.log(f"Stamp with value {value} generated in {RNS.prettytime(duration)}, {rounds} rounds, {RNS.prettysize(speed, suffix='rps')}", RNS.LOG_DEBUG) if RNS.sl(RNS.LOG_DEBUG) else None

    return stamp, value

def cancel_work(message_id):
    if RNS.vendor.platformutils.is_windows() or RNS.vendor.platformutils.is_darwin():
        try:
            if message_id in active_jobs:
                active_jobs[message_id] = True

        except Exception as e:
            RNS.log("Error while terminating stamp generation workers: {e}", RNS.LOG_ERROR)
            RNS.trace_exception(e)

    elif RNS.vendor.platformutils.is_android():
        try:
            if message_id in active_jobs:
                active_jobs[message_id] = True

        except Exception as e:
            RNS.log("Error while terminating stamp generation workers: {e}", RNS.LOG_ERROR)
            RNS.trace_exception(e)

    else:
        try:
            if message_id in active_jobs:
                stop_event = active_jobs[message_id][0]
                result_queue = active_jobs[message_id][1]
                stop_event.set()
                result_queue.put(None)
                active_jobs.pop(message_id)

        except Exception as e:
            RNS.log("Error while terminating stamp generation workers: {e}", RNS.LOG_ERROR)
            RNS.trace_exception(e)

def job_simple(stamp_cost, workblock, message_id, size):
    # A simple, single-process stamp generator.
    # should work on any platform, and is used
    # as a fall-back, in case of limited multi-
    # processing and/or acceleration support.

    rounds = 0
    pstamp = os.urandom(size)
    st = time.time()

    active_jobs[message_id] = False

    target = 0b1<<256-stamp_cost
    c = 0

    while True:
        c += 1
        if c >= STOP_CHECK_INTERVAL:
            c = 0
            if active_jobs[message_id]: break

        if int.from_bytes(RNS.Identity.full_hash(workblock+pstamp), byteorder="big") <= target: break
        pstamp = os.urandom(size); rounds += 1

        if rounds % 250000 == 0:
            speed = rounds / max(time.time()-st, 1E-6)
            RNS.log(f"Stamp generation running. {rounds} rounds completed so far, {RNS.prettysize(speed, suffix='')} rounds per second", RNS.LOG_DEBUG) if RNS.sl(RNS.LOG_DEBUG) else None

    cancelled = (active_jobs[message_id] == True)
    active_jobs.pop(message_id)
    return (None if cancelled else pstamp), rounds

def job_linux_managed(stamp_cost, workblock, message_id, size):
    ctx = multiprocessing.get_context("fork")
    cores = multiprocessing.cpu_count()
    jobs = cores if cores <= 12 else int(cores/2)
    
    allow_kill = True
    stamp = None
    total_rounds = 0
    
    with worker_context(ctx, stamp_cost, workblock, message_id) as (stop_event, result_queue, rounds_queue, job_procs):
        RNS.log(f"Starting {jobs} stamp generation workers", RNS.LOG_DEBUG)
        
        def job(stop_event, sc, wb, pn):
            rounds = 0; pstamp = os.urandom(size);
            target = 0b1<<256 - sc; c = 0

            while True:
                c += 1
                if c >= STOP_CHECK_INTERVAL:
                    c = 0
                    if stop_event.is_set(): break

                m = wb+pstamp
                if int.from_bytes(RNS.Identity.full_hash(m), byteorder="big") <= target: break
                pstamp = os.urandom(size); rounds += 1

            if not stop_event.is_set():
                stop_event.set()
                try: result_queue.put_nowait(pstamp)
                except: pass
            
            try: rounds_queue.put_nowait(rounds)
            except: pass

        for jpn in range(jobs):
            p = ctx.Process(target=job, args=(stop_event, stamp_cost, workblock, jpn), daemon=True)
            job_procs.append(p)
            p.start()
        
        try: stamp = result_queue.get(timeout=None)
        except Exception as e:
            RNS.log(f"Failed to get result from workers: {e}", RNS.LOG_ERROR)
            stamp = None
        
        # Collect any potential spurious
        # results from worker queue.
        try:
            while True: result_queue.get_nowait()
        except: pass

        for j in range(jobs):
            nrounds = 0
            try:
                nrounds = rounds_queue.get(timeout=2)
            except Exception as e:
                RNS.log(f"Failed to get round stats part {j}: {e}", RNS.LOG_ERROR)
            total_rounds += nrounds

        all_exited = False
        exit_timeout = time.time() + 5
        while time.time() < exit_timeout:
            if not any(p.is_alive() for p in job_procs):
                all_exited = True
                break
            time.sleep(0.005)

        if not all_exited:
            RNS.log("Stamp generation IPC timeout, possible worker deadlock. Terminating remaining processes.", RNS.LOG_ERROR)
            if allow_kill:
                for j in range(jobs):
                    process = job_procs[j]
                    process.kill()
            else:
                return None

        else:
            for j in range(jobs):
                process = job_procs[j]
                process.join()
                # RNS.log(f"Joined {j} / {process}", RNS.LOG_DEBUG) # TODO: Remove

    return stamp, total_rounds

def job_linux(stamp_cost, workblock, message_id, size):
    allow_kill = True
    stamp = None
    total_rounds = 0
    cores = multiprocessing.cpu_count()
    jobs = cores if cores <= 12 else int(cores/2)
    stop_event   = multiprocessing.Event()
    result_queue = multiprocessing.Queue(1)
    rounds_queue = multiprocessing.Queue()

    def job(stop_event, pn, sc, wb):
        rounds = 0; pstamp = os.urandom(size);
        target = 0b1<<256 - sc; c = 0

        while True:
            c += 1
            if c >= STOP_CHECK_INTERVAL:
                c = 0
                if stop_event.is_set(): break

            m = wb+pstamp
            if int.from_bytes(RNS.Identity.full_hash(m), byteorder="big") <= target: break
            pstamp = os.urandom(size); rounds += 1

        if not stop_event.is_set():
            stop_event.set()
            result_queue.put(pstamp)

        rounds_queue.put(rounds)
    
    job_procs = []
    RNS.log(f"Starting {jobs} stamp generation workers", RNS.LOG_DEBUG)
    for jpn in range(jobs):
        process = multiprocessing.get_context("fork").Process(target=job, kwargs={"stop_event": stop_event, "pn": jpn, "sc": stamp_cost, "wb": workblock}, daemon=True)
        job_procs.append(process)
        process.start()

    active_jobs[message_id] = [stop_event, result_queue]

    try:
        stamp = result_queue.get()

        # Collect any potential spurious
        # results from worker queue.
        try:
            while True: result_queue.get_nowait()
        except: pass

        for j in range(jobs):
            nrounds = 0
            try:
                nrounds = rounds_queue.get(timeout=2)
            except Exception as e:
                RNS.log(f"Failed to get round stats part {j}: {e}", RNS.LOG_ERROR)
            total_rounds += nrounds

        all_exited = False
        exit_timeout = time.time() + 5
        while time.time() < exit_timeout:
            if not any(p.is_alive() for p in job_procs):
                all_exited = True
                break
            time.sleep(0.005)

        if not all_exited:
            RNS.log("Stamp generation IPC timeout, possible worker deadlock. Terminating remaining processes.", RNS.LOG_ERROR)
            if allow_kill:
                for j in range(jobs):
                    process = job_procs[j]
                    process.kill()
            else:
                return None

        else:
            for j in range(jobs):
                process = job_procs[j]
                process.join()

    except Exception as e: RNS.log(f"Error while getting stamp generation result: {e}", RNS.LOG_ERROR)
    finally: active_jobs.pop(message_id, None)

    return stamp, total_rounds

def job_android(stamp_cost, workblock, message_id, size):
    # Semaphore support is flaky to non-existent on
    # Android, so we need to manually dispatch and
    # manage workloads here, while periodically
    # checking in on the progress.
    
    stamp = None
    start_time = time.time()
    total_rounds = 0
    rounds_per_worker = 1000
    
    use_nacl = False
    try:
        import nacl.encoding
        import nacl.hash
        use_nacl = True
    except: pass

    if use_nacl:
        def full_hash(m): return nacl.hash.sha256(m, encoder=nacl.encoding.RawEncoder)
    else:
        def full_hash(m): return RNS.Identity.full_hash(m)

    wm = multiprocessing.Manager()
    jobs = multiprocessing.cpu_count()

    def job(procnum=None, results_dict=None, wb=None, sc=None, jr=None):
        try:
            rounds = 0
            found_stamp = None
            target = 0b1<<256-sc
            fh = full_hash

            while True:
                pstamp = os.urandom(size)
                rounds += 1
                if int.from_bytes(fh(wb+pstamp), byteorder="big") <= target:
                    found_stamp = pstamp; break
                if rounds >= jr: break

            results_dict[procnum] = [found_stamp, rounds]

        except Exception as e:
            RNS.log(f"Stamp generation worker error: {e}", RNS.LOG_ERROR)
            RNS.trace_exception(e)

    active_jobs[message_id] = False;

    RNS.log(f"Dispatching {jobs} workers for stamp generation...", RNS.LOG_DEBUG) if RNS.sl(RNS.LOG_DEBUG) else None

    results_dict = wm.dict()
    while stamp == None and active_jobs[message_id] == False:
        job_procs = []
        try:
            for pnum in range(jobs):
                pargs = {"procnum":pnum, "results_dict": results_dict, "wb": workblock, "sc":stamp_cost, "jr":rounds_per_worker}
                process = multiprocessing.Process(target=job, kwargs=pargs)
                job_procs.append(process)
                process.start()

            for process in job_procs:
                process.join()

            for j in results_dict:
                r = results_dict[j]
                total_rounds += r[1]
                if r[0] != None:
                    stamp = r[0]

            if stamp == None:
                elapsed = max(time.time()-start_time, 1E-6)
                speed = total_rounds/elapsed
                RNS.log(f"Stamp generation running. {total_rounds} rounds completed so far, {RNS.prettysize(speed, suffix='rps')}", RNS.LOG_DEBUG) if RNS.sl(RNS.LOG_DEBUG) else None
        
        except Exception as e:
            RNS.log(f"Stamp generation job error: {e}", RNS.LOG_ERROR)
            RNS.trace_exception(e)

    active_jobs.pop(message_id)

    return stamp, total_rounds

###########################
# Persistence and loading #
###########################

DESTINATION_STAMP_EXTENSION = ".dstamp"
STAMP_FILE_VERSION          = 1

def stamps_dir(create=False):
    storagepath = RNS.Reticulum.storagepath
    if not storagepath: return None
    path = os.path.join(storagepath, "stamps")
    if create and not os.path.isdir(path):
        try: os.makedirs(path, exist_ok=True)
        except Exception as e:
            RNS.log(f"An error occurred while creating destination stamp storage path: {e}", RNS.LOG_WARNING)
            return None
    return path

def destination_stamp_file(destination_hash, create=False):
    dir_path = stamps_dir(create=create)
    if dir_path == None: return None
    return os.path.join(dir_path, RNS.hexrep(destination_hash, delimit=False)+DESTINATION_STAMP_EXTENSION)

def store_destination_stamp(destination_hash, stamp, stamp_value):
    file_path = destination_stamp_file(destination_hash, create=True)
    if file_path == None:
        RNS.log("No storage path available, destination stamp not persisted", RNS.LOG_WARNING)
        return False

    try:
        data = msgpack.packb({"v": STAMP_FILE_VERSION, "stamp":   stamp,
                              "value":   stamp_value, "generated": time.time()})

        temp_path = file_path+f".{os.getpid()}.tmp"
        with open(temp_path, "wb") as temp_file: temp_file.write(data)
        os.replace(temp_path, file_path)
        return True

    except Exception as e:
        RNS.log(f"An error occurred while persisting destination stamp: {e}", RNS.LOG_WARNING)
        return False

def load_destination_stamp(destination_hash, material, required_value):
    file_path = destination_stamp_file(destination_hash)
    if file_path == None or not os.path.isfile(file_path): return None

    try:
        with open(file_path, "rb") as stamp_file: entry = msgpack.unpackb(stamp_file.read())
        if entry.get("v") != STAMP_FILE_VERSION: return None
        workblock = stamp_workblock(material, expand_rounds=0)

        stamp = entry["stamp"]
        if not isinstance(stamp, bytes) or len(stamp) != RNS.Identity.STAMPLENGTH:
            RNS.log(f"Stored stamp for {RNS.prettyhexrep(destination_hash)} has invalid format. Regenerating on next announce.", RNS.LOG_DEBUG) if RNS.sl(RNS.LOG_DEBUG) else None
            return None

        value = stamp_value(workblock, stamp)
        if value < required_value:
            RNS.log(f"Stored stamp for {RNS.prettyhexrep(destination_hash)} has value {value}, below required {required_value}. Regenerating on next announce.", RNS.LOG_DEBUG) if RNS.sl(RNS.LOG_DEBUG) else None
            return None

        return (stamp, value)

    except Exception as e:
        RNS.log(f"Ignoring unreadable destination stamp file for {RNS.prettyhexrep(destination_hash)}: {e}", RNS.LOG_WARNING)
        return None
