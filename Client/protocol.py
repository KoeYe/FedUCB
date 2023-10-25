import struct
import zlib
import torch

class Protocol:
    def __init__(self):
        # header和footer用于标记消息的开始和结束
        self.header = b"<MESSAGE>"
        self.footer = b"<END>"
        # 用于查找消息的类型
        self.msg_types = {'params': b'\x01', 'grad': b'\x02', 'done': b'\x03', 'text': b'\x04'}
        # 用于反向查找消息类型
        self.inverse_msg_types = {v: k for k, v in self.msg_types.items()}
        # 用于查找精度的类型
        self.dtype_map = {torch.float32: ('f', 4), torch.float64: ('d', 8)}
        # 用于反向查找精度类型
        self.size_to_dtype = {v[1]: k for k, v in self.dtype_map.items()}

    def encode_text(self, text, msg_type, compress=False):
        if msg_type not in self.msg_types:
            raise ValueError("Invalid message type")
        # 定义header
        header = self.header + self.msg_types[msg_type]
        payload = b''
        # 将text转换为bytes
        text = text.encode()
        # 如果需要压缩，则对text进行压缩
        if compress:
            text = zlib.compress(text)
        # 将header、payload和footer拼接成一个完整的消息
        message = header + text + self.footer
        length = len(message)
        message = struct.pack('i', length) + message
        return message

    def encode_tensor(self, tensors, msg_type, compress=False):
        if msg_type not in self.msg_types:
            raise ValueError("Invalid message type")

        if not all(torch.is_tensor(tensor) for tensor in tensors):
            raise ValueError("All inputs must be torch tensors")

        # 定义header
        header = self.header + self.msg_types[msg_type]
        payload = b''

        # 将tensor的精度和维度信息序列化后与tensor数据一起加入到payload中
        for tensor in tensors:
            if tensor.dtype not in self.dtype_map:
                raise ValueError("Unsupported tensor dtype")
            dtype_char, dtype_size = self.dtype_map[tensor.dtype]
            dims = len(tensor.shape)
            tensor_header = struct.pack('ii', dtype_size, dims)
            tensor_header += struct.pack('i' * dims, *tensor.shape)
            tensor_data = tensor.detach().numpy().tobytes()
            payload += tensor_header + tensor_data

        # 如果需要压缩，则对payload进行压缩
        if compress:
            payload = zlib.compress(payload)

        # 将header、payload和footer拼接成一个完整的消息
        message = header + payload + self.footer
        length = len(message)
        message = struct.pack('i', length) + message
        return message

    def decode(self, message, decompress=False):
        # 自动判断消息类型
        msg_type = message[len(self.header):len(self.header)+1]
        if msg_type not in self.inverse_msg_types:
            raise ValueError("Unknown message type")
        msg_type = self.inverse_msg_types[msg_type]
        if msg_type == 'text':
            return self.decode_text(message, decompress)
        elif msg_type == 'done':
            return msg_type, None
        else:
            return self.decode_tensor(message, decompress)

    def decode_text(self, message, decompress=False):
        if not message.startswith(self.header) or not message.endswith(self.footer):
            raise ValueError("Invalid message format")
        # 去掉header和footer
        message = message[len(self.header):-len(self.footer)]
        # 获取消息类型，因为消息类型是一个字节，所以取第一个字节即可
        msg_type = message[:1]
        if msg_type not in self.inverse_msg_types:
            raise ValueError("Unknown message type")
        # 利用反向映射，将消息类型转换为字符串
        msg_type = self.inverse_msg_types[msg_type]
        # 去掉消息类型剩下的全是payload
        payload = message[1:]
        # 如果需要解压，则对payload进行解压
        if decompress:
            payload = zlib.decompress(payload)
        # 将payload转换为字符串
        text = payload.decode()
        # 将text转化成str
        text = str(text)
        return msg_type, text


    def decode_tensor(self, message, decompress=False):
        print(message)
        if not message.startswith(self.header) or not message.endswith(self.footer):
            raise ValueError("Invalid message format")

        # 去掉header和footer
        message = message[len(self.header):-len(self.footer)]
        # 获取消息类型，因为消息类型是一个字节，所以取第一个字节即可
        msg_type = message[:1]
        if msg_type not in self.inverse_msg_types:
            raise ValueError("Unknown message type")
        # 利用反向映射，将消息类型转换为字符串
        msg_type = self.inverse_msg_types[msg_type]

        # 去掉消息类型剩下的全是payload
        payload = message[1:]

        # 如果需要解压，则对payload进行解压
        if decompress:
            payload = zlib.decompress(payload)

        # 从payload中解析出tensor的精度和维度信息
        tensors = []
        while payload:
            # 从payload中解析出tensor的精度和维度信息，也就是前8个字节
            dtype_size, dims = struct.unpack('ii', payload[:8])
            # 去掉前8个字节，剩下的就是tensor的shape+tensor的数据+其他的tensor
            payload = payload[8:]
            # 根据dim从payload中解析出tensor的维度信息
            tensor_shape = struct.unpack('i' * dims, payload[:4*dims])
            # 去掉维度信息，剩下的就是tensor的数据+其他的tensor
            payload = payload[4*dims:]

            # 根据dtype_size从payload中解析出tensor的精度信息，并处理精度不支持的情况
            if dtype_size not in self.size_to_dtype:
                raise ValueError("Unknown dtype size")
            dtype = self.size_to_dtype[dtype_size]
            dtype_char, _ = self.dtype_map[dtype]

            # 根据tensor的精度和维度信息，从payload中解析出tensor的数据
            # 这里的num_elements是tensor的元素个数
            num_elements = int(torch.tensor(tensor_shape).prod())
            # 这里的tensor_data是一个元组，元组中的每个元素都是tensor的一个元素
            tensor_data = struct.unpack(f'{num_elements}{dtype_char}', payload[:num_elements*dtype_size])
            # 将tensor_data转换为tensor
            tensor = torch.tensor(tensor_data, dtype=dtype).reshape(tensor_shape)
            # 将tensor加入到tensors中
            tensors.append(tensor)
            # 去掉tensor的数据，剩下的就是其他的tensor
            payload = payload[num_elements*dtype_size:]

        return msg_type, tensors

    def encode_gradients(self, params, msg_type, compress=False):
        # 这里直接将params中的梯度取出来，因为params本身的元素就是parameter，parameter有grad属性
        gradients = [p.grad for p in params if p.grad is not None]
        return self.encode_tensor(gradients, msg_type, compress)

    def encode_parameters(self, params, msg_type, compress=False):
        # 这里params本身的元素就是parameter
        parameters = [p for p in params]
        return self.encode_tensor(parameters, msg_type, compress)

    def decode_gradients(self, message, decompress=False):
        msg_type, gradients = self.decode_tensor(message, decompress)
        if msg_type != 'grad':
            raise ValueError("Invalid message type for gradients")
        return gradients

    def decode_parameters(self, message, decompress=False):
        msg_type, parameters = self.decode_tensor(message, decompress)
        if msg_type != 'params':
            raise ValueError("Invalid message type for parameters")
        return parameters

    def encode_done(self, compress=False):
        return self.encode_text('', 'done', compress)

def test():
    # 测试Protocol类，自动输出测试结果
    protocol = Protocol()
    text = "Hello World!"
    message = protocol.encode_text(text, 'text')
    # 预测结果应该是
    message_should_be = b'<MESSAGE>\x04Hello World!<END>'
    # 测试结果
    print(message == message_should_be)
    # 预测结果应该是
    text_should_be = text
    # 测试结果
    print(protocol.decode_text(message)[1] == text_should_be)

    # 还有其他的测试，测encode_tensor和decode_tensor，encode_gradients和decode_gradients，encode_parameters和decode_parameters
    # 注意精度只能是float32和float64，其他的精度都不支持
    # 测试encode_tensor和decode_tensor
    import numpy as np
    import torch

    tensor = torch.tensor(np.random.randn(2, 3))
    message = protocol.encode_tensor([tensor], 'params')
    # 预测结果应该是
    tensor_should_be = tensor
    # 测试结果
    print(protocol.decode_tensor(message)[1][0] == tensor_should_be)

    # 测试encode_gradients和decode_gradients
    tensor = torch.tensor(np.random.randn(2, 3), requires_grad=True)
    tensor.sum().backward()
    message = protocol.encode_gradients([tensor], 'grad')
    # 预测结果应该是
    tensor_should_be = tensor.grad
    # 测试结果
    print(protocol.decode_gradients(message)[0] == tensor_should_be)

    # 测试encode_parameters和decode_parameters
    tensor = torch.tensor(np.random.randn(2, 3), requires_grad=True)
    message = protocol.encode_parameters([tensor], 'params')
    # 预测结果应该是
    tensor_should_be = tensor
    # 测试结果
    print(protocol.decode_parameters(message)[0] == tensor_should_be)

    # 测试encode_done
    message = protocol.encode_done()
    # 预测结果应该是
    message_should_be = b'<MESSAGE>\x03<END>'
    # 测试结果
    print(message == message_should_be)

    class testModel(torch.nn.Module):
        def __init__(self):
            super(testModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = torch.nn.Dropout2d()
            self.fc1 = torch.nn.Linear(320, 50)
            self.fc2 = torch.nn.Linear(50, 10)

    test_model = testModel()
    client_model = testModel()
    params_value = [param.data for param in test_model.parameters()]
    message = protocol.encode_parameters(params_value, 'params')
    _, result = protocol.decode(message)
    # 将result中的tensor给到client的model中
    result_dict = {}
    # print(len(result))
    for i in client_model.state_dict().keys():
        result_dict[i] = result.pop(0)

    # print(result_dict)
    client_model.load_state_dict(result_dict)
    client_model_params_value = [param.data for param in client_model.parameters()]

    # 测试结果
    for i in range(len(params_value)):
        print(torch.equal(params_value[i], client_model_params_value[i]))


if __name__ == '__main__':
    test()