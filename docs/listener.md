## listener.py Overview

`listener.py` is a simple script to debug incoming messages on the same Redis channel:

```python
import redis
import time

def main():
    r = redis.Redis(host='localhost', port=6379, db=0)
    pubsub = r.pubsub()
    pubsub.subscribe('agent:queen')
    print("Subscribed to channel: agent:queen")

    while True:
        message = pubsub.get_message()
        if message:
            print(message)
        time.sleep(0.05)

if __name__ == '__main__':
    main()
```

* **Purpose**: Print every raw message published to `agent:queen`.
* **Run** with:

  ```bash
  python listener.py
  ```

---

## Usage

1. **Ensure Redis is running** on `localhost:6379`.
2. **Activate your virtual environment**:

   ```bash
   source .venv/bin/activate
   ```
3. **Run the Queen Agent**:

   ```bash
   python agent.py
   ```
4. **Optionally, run the listener** (for debugging):

   ```bash
   python listener.py
   ```
5. **Publish a test message** (via `redis-cli`):

   ```bash
   redis-cli PUBLISH agent:queen \
     '{"type":"user","content":"what\'s the price of BTC","channel_id":"agent:queen","metadata":{"model":"claude-3-5-haiku-latest"}}'
   ```
