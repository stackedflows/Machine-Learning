#python Inference.py
import torch

from MNIST import Net

def main():
    #train and load model
    pytorch_model = Net()
    pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
    pytorch_model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    #recommended barracuda export settings
    torch.onnx.export(pytorch_model,
                        dummy_input,
                        'MNIST.onnx',
                        export_params = True,
                        opset_version = 9,
                        do_constant_folding = True,
                        input_names = ['X'],
                        output_names = ['Y'],
                        verbose = True
                     )
    
if __name__ == '__main__':
    main()
