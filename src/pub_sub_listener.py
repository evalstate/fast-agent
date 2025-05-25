import redis
import time

def main():
    # Connect to Redis (adjust host/port/db as needed)
    r = redis.Redis(host='localhost', port=6379, db=0)

    # Create a PubSub object and subscribe to the channel
    pubsub = r.pubsub()
    channel_name = 'agent:queen'
    pubsub.subscribe(channel_name)
    print(f"Subscribed to channel: {channel_name}")

    # Loop forever, polling for new messages
    while True:
        message = pubsub.get_message()
        if message:
            # Print the raw message dict
            print(message)
        # Sleep briefly to avoid busy‑waiting
        time.sleep(0.05)

if __name__ == '__main__':
    main()
