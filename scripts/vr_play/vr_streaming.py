import numpy as np

# from sock import SocketClient
from controller import Controller
import zmq
import signal
import pickle
import loguru

# NOTE This is the ip and port of the pc host connected to vr
GLOBAL_IP = "192.168.1.109"
GLOBAL_PORT = "34567"


logger = loguru.logger


def main():
    context = zmq.Context()
    sock = context.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(f"tcp://{GLOBAL_IP}:{GLOBAL_PORT}")

    def sig_handler(sig, frame):
        logger.info("Exit")
        sock.close()
        context.term()
        exit(0)

    signal.signal(signal.SIGINT, sig_handler)

    print("Waiting for VR connection...")

    controller = Controller()
    print("VR connection established. Streaming data...")
    actions = controller.get_action()
    print("Streaming data from VR controllers...")
    for idx, data in enumerate(actions):
        bin_data = pickle.dumps(data)
        print(f"Sending data packet {idx+1}: {data}")
        sock.send(bin_data)
        logger.info(f"Sent data packet {idx+1}")
        print(f"Sent data packet {idx+1}: {data}")



if __name__ == '__main__':
    # test()
    main()
