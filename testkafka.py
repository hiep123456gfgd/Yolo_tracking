from multiprocessing import Process
import os

def run_producer():
    os.system("python Producer.py")

def run_consumer():
    os.system("python consumer1.py")

if __name__ == "__main__":
    producer_process = Process(target=run_producer)
    consumer_process = Process(target=run_consumer)

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
