# cli.py
"""
CLI entry point + live dashboard for vLLM inference server.
Modes: interactive, demo, single-shot.
"""
import argparse
import sys
import time
import threading

from server import InferenceServer


def format_dashboard(stats, uptime=0, throughput=0):
    """Render ASCII dashboard."""
    s = stats['scheduler']
    lines = []
    lines.append("╔══════════════ vLLM-Inference Server ══════════════╗")
    lines.append(f"│ [Queue: {s['pending']}]  [Active: {s['active']}]  [Pages: {s['pages_used']}/{s['pages_used']+s['pages_free']}]  │")
    lines.append(f"│ [Throughput: {throughput:.1f} tok/s]  [Uptime: {uptime:.1f}s]            │")
    lines.append("└───────────────────────────────────────────────────┘")
    return '\n'.join(lines)


def dashboard_thread(server, stop_event):
    """Background thread that refreshes the dashboard."""
    start = time.time()
    total_tokens = 0
    while not stop_event.is_set():
        stats = server.stats()
        elapsed = time.time() - start
        throughput = total_tokens / elapsed if elapsed > 0 else 0
        # Clear screen and redraw
        output = format_dashboard(stats, elapsed, throughput)
        sys.stdout.write('\033[2J\033[H' + output + '\n')
        sys.stdout.flush()
        # Count completed tokens
        total_tokens = stats['scheduler'].get('completed', 0) * 10  # rough estimate
        time.sleep(0.2)


def demo_mode(server, prompts):
    """Run demo: submit multiple prompts, show continuous batching."""
    print("Starting demo mode with prompts:", prompts)
    results = {}
    ids = []
    for p in prompts:
        rid = server.submit_async(p, max_new_tokens=20, temperature=0.8, top_k=20)
        ids.append((rid, p))
        print(f"  Submitted Req#{rid}: {p}")
        time.sleep(0.1)  # Stagger submissions to show continuous batching

    for rid, prompt in ids:
        result = server.get_result(rid, timeout=60)
        results[rid] = (prompt, result)
        print(f"\n  Req#{rid} [{prompt}]: {result}")

    return results


def interactive_mode(server):
    """Interactive REPL."""
    print("vLLM Inference Server — Interactive Mode")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue
            result = server.generate(prompt, max_new_tokens=30, temperature=0.8, top_k=20, timeout=60)
            print(f"\n{result}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="vLLM-Inspired Inference Server")
    parser.add_argument('--model', default='gpt_chinese.npz', help='Model path')
    parser.add_argument('--model-url', default=None, help='Model download URL (auto-download if model missing)')
    parser.add_argument('--batch-size', type=int, default=8, help='Max batch size')
    parser.add_argument('--workers', type=int, default=4, help='Thread pool size')
    parser.add_argument('--pages', type=int, default=256, help='Number of KV pages')
    parser.add_argument('--window-ms', type=int, default=50, help='Scheduling window (ms)')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--prompt', type=str, help='Single prompt mode')
    args = parser.parse_args()

    # Ensure model exists (download if needed)
    from model_forward import ensure_model
    try:
        model_path = ensure_model(args.model, url=args.model_url)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    server = InferenceServer(
        model_path=model_path,
        max_batch_size=args.batch_size,
        num_workers=args.workers,
        window_ms=args.window_ms,
        max_pages=args.pages,
    )
    server.start()
    print(f"Server started: batch={args.batch_size}, workers={args.workers}, pages={args.pages}")

    try:
        if args.prompt:
            result = server.generate(args.prompt, max_new_tokens=30, temperature=0.8, top_k=20, timeout=60)
            print(result)
        elif args.demo:
            prompts = ["春眠不觉晓", "大江东去", "人生若只如初见", "人工智能", "明月几时有"]
            demo_mode(server, prompts)
        else:
            interactive_mode(server)
    finally:
        server.shutdown()
        print("\nServer shutdown.")


if __name__ == "__main__":
    main()
