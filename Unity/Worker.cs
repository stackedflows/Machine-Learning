using System;
using System.Collections;
using UnityEngine;
using Unity.Barracuda;

namespace MNISTPainter
{
    public class Worker : IDisposable
    {
        private IWorker worker;
        private Model model;
        private float[] results;
        private PrecompiledComputeOps ops;

        public Worker(NNModel nnmodel, WorkerFactory.Type type)
        {
            bool verbose = false;
            model = ModelLoader.Load(nnmodel, verbose);
            worker = WorkerFactory.CreateWorker(type, model, verbose);

            var kernels = ComputeShaderSingleton.Instance.kernels;
            ops = new PrecompiledComputeOps(kernels, kernels[0]);
        }

        public void Dispose()
        {
            worker?.Dispose();
            model = null;
            ops = null;
        }

        public IEnumerator ExecuteAsync(Texture2D texture)
        {
            int width = texture.width;
            int height = texture.height;

            Tensor input = new Tensor(1, height, width, 1);

            for(int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    input[0, 27 - y, x, 0] = texture.GetPixel(x, y).r;
                }
                yield return worker.ExecuteAsync(input);
            }

            Tensor output = worker.PeekOutput();

            results = output.data.Download(output.shape);

            input.Dispose();
            output.Dispose();
        }

        public IEnumerator ExecuteAsyncTexture(Texture2D texture)
        {
            Tensor input = new Tensor(texture, 1);
            yield return worker.ExecuteAsync(input);

            Tensor output = worker.PeekOutput();
            results = output.data.Download(output.shape);

            input.Dispose();
            output.Dispose();
        }

        public float[] get_result()
        {
            return results;
        }

        public TensorShape GetInputShape(int index)
        {
            return new TensorShape(model.inputs[index].shape);
        }
    }

}
