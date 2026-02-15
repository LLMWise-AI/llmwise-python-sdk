from llmwise import LLMWise


def main() -> None:
    client = LLMWise("mm_sk_...")  # or set LLMWISE_API_KEY env var
    by_model: dict[str, str] = {}

    for ev in client.compare_stream(
        models=["gpt-5.2", "claude-sonnet-4.5", "gemini-3-flash"],
        messages=[{"role": "user", "content": "Give 3 growth ideas for an AI gateway."}],
    ):
        if ev.get("event") == "summary":
            print("\nsummary:", ev)
            continue
        if ev.get("event") == "done":
            print("\ndone:", ev)
            break
        if ev.get("delta"):
            by_model.setdefault(ev["model"], "")
            by_model[ev["model"]] += ev["delta"]

    print("\n--- outputs ---")
    for mid, text in by_model.items():
        print("\n#", mid)
        print(text)


if __name__ == "__main__":
    main()
