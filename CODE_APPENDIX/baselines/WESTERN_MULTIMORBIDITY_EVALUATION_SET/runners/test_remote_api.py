from __future__ import annotations

import argparse
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=120,
        max_retries=0,
    )

    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": '{"ping": "hello"}'},
        ],
        temperature=0.0,
        max_tokens=64,
    )

    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
