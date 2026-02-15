from llmwise import LLMWise


def main() -> None:
    client = LLMWise("mm_sk_...")  # or set LLMWISE_API_KEY env var
    resp = client.chat(
        model="auto",
        messages=[{"role": "user", "content": "Write a 1-sentence startup tagline for an LLM load balancer."}],
        stream=False,
    )
    print(resp["content"])


if __name__ == "__main__":
    main()
