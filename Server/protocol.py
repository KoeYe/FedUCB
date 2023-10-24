import struct
import zlib
import torch

class Protocol:
    def __init__(self):
        # header和footer用于标记消息的开始和结束
        self.header = b"<TENSOR>"
        self.footer = b"<END>"
        # 用于查找消息的类型
        self.msg_types = {'params': b'\x01', 'grad': b'\x02'}
        # 用于反向查找消息类型
        self.inverse_msg_types = {v: k for k, v in self.msg_types.items()}
        # 用于查找精度的类型
        self.dtype_map = {torch.float32: ('f', 4), torch.float64: ('d', 8)}
        # 用于反向查找精度类型
        self.size_to_dtype = {v[1]: k for k, v in self.dtype_map.items()}

    def encode(self, tensors, msg_type, compress=False):
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
        return message

    def decode(self, message, decompress=False):
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
        return self.encode(gradients, msg_type, compress)

    def encode_parameters(self, params, msg_type, compress=False):
        # 这里params本身的元素就是parameter
        parameters = [p for p in params]
        return self.encode(parameters, msg_type, compress)

    def decode_gradients(self, message, decompress=False):
        msg_type, gradients = self.decode(message, decompress)
        if msg_type != 'grad':
            raise ValueError("Invalid message type for gradients")
        return gradients

    def decode_parameters(self, message, decompress=False):
        msg_type, parameters = self.decode(message, decompress)
        if msg_type != 'params':
            raise ValueError("Invalid message type for parameters")
        return parameters
