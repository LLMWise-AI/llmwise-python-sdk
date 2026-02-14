from llmwise_sdk import LLMWise


def main() -> None:
    client = LLMWise("mm_sk_...")  # or set LLMWISE_API_KEY env var
    for ev in client.chat_stream(
        model="claude-sonnet-4.5",
        messages=[{"role": "user", "content": "Explain what a circuit breaker is in 5 bullets."}],
    ):
        if ev.get("event") == "done":
            print("\n\nDONE", ev)
            break
        if ev.get("delta"):
            print(ev["delta"], end="", flush=True)


if __name__ == "__main__":
    main()
