# server.py
"""
Multi-threaded inference server.
Scheduler thread runs continuous batching loop.
Main thread accepts requests and returns results.
"""
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

from scheduler import ContinuousBatchScheduler

class InferenceServer:
    def __init__(self, model_path, max_batch_size=8, num_workers=4, window_ms=50, max_pages=256):
        self.model_path = model_path
        self.scheduler = ContinuousBatchScheduler(
            max_batch_size=max_batch_size,
            window_ms=window_ms,
            max_pages=max_pages,
        )
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.num_workers = num_workers
        self._running = False
        self._params = None

    def start(self):
        self._params, _, _ = self.scheduler.load_model(self.model_path)
        self._running = True
        self._sched_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._sched_thread.start()

    def _scheduler_loop(self):
        while self._running:
            # Drain incoming requests
            try:
                while True:
                    req = self.request_queue.get_nowait()
                    self.scheduler.submit(**req)
            except queue.Empty:
                pass

            # Run one decode step
            if self.scheduler.has_work():
                completed = self.scheduler.step(self._params)
                for req_id, text in completed.items():
                    self.result_queue.put((req_id, text))
            else:
                time.sleep(0.005)  # 5ms idle sleep

    def generate(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50, timeout=120):
        req_id = self.submit_async(prompt, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_k=top_k)
        return self.get_result(req_id, timeout=timeout)

    def submit_async(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
        # Directly submit to scheduler (thread-safe via scheduler's internal lock)
        req_id = self.scheduler.submit(prompt, max_new_tokens=max_new_tokens,
                                       temperature=temperature, top_k=top_k)
        return req_id

    def get_result(self, req_id, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                rid, text = self.result_queue.get(timeout=min(0.1, deadline - time.time()))
                if rid == req_id:
                    return text
                # Not our result — check scheduler.completed dict
                if req_id in self.scheduler.completed:
                    return self.scheduler.completed.pop(req_id)
            except queue.Empty:
                # Check if already completed
                if req_id in self.scheduler.completed:
                    return self.scheduler.completed.pop(req_id)
                continue
        raise TimeoutError(f"Request {req_id} timed out after {timeout}s")

    def shutdown(self):
        self._running = False
        if hasattr(self, '_sched_thread'):
            self._sched_thread.join(timeout=5)

    def stats(self):
        return {
            'scheduler': self.scheduler.stats(),
            'running': self._running,
        }
