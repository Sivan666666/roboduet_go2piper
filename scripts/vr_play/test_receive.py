import zmq
import pickle

context = zmq.Context()
sock = context.socket(zmq.PULL)
sock.connect("tcp://192.168.1.126:34567")

while True:
    data = sock.recv()
    action = pickle.loads(data)
    print("收到数据:", action[:7])