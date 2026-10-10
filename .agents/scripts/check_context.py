import sys
import json
import os

def estimate_tokens_from_transcript(transcript_path):
    if not transcript_path or not os.path.exists(transcript_path):
        return 0
    try:
        # A transcript.jsonl includes JSON structure + text. 
        # Standard ratio for JSONL conversation logs to LLM token consumption is ~3.6 - 3.8 bytes/token.
        file_size = os.path.getsize(transcript_path)
        return int(file_size / 3.6)
    except Exception:
        return 0

def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"injectSteps": []}))
            return

        payload = json.loads(raw_input)
        transcript_path = payload.get("transcriptPath", "")

        # Thresholds
        SMART_ZONE_LIMIT = 140000
        tokens = estimate_tokens_from_transcript(transcript_path)

        if tokens >= SMART_ZONE_LIMIT:
            msg = (
                f"🚨 SMART ZONE CEILING REACHED (~{tokens:,} tokens estimated).\n"
                f"Model reasoning, multi-step instruction adherence, and subtle bug detection degrade past ~140k tokens.\n\n"
                f"ACTION REQUIRED:\n"
                f"1. Inform the user that the session has reached the 140k Smart Zone boundary.\n"
                f"2. Run the 'smart-zone-handoff' skill to write a complete, high-fidelity handoff dossier to .scratch/handoffs/latest.md.\n"
                f"3. Advise the user to start a fresh chat session and resume via: '@.scratch/handoffs/latest.md continue'."
            )
            output = {
                "injectSteps": [
                    {
                        "ephemeralMessage": msg
                    }
                ]
            }
        else:
            output = {"injectSteps": []}

        print(json.dumps(output))
    except Exception as e:
        # Never crash the invocation loop if monitoring fails
        print(json.dumps({"injectSteps": []}))

if __name__ == "__main__":
    main()
