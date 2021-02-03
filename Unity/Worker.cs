using System;
using System.Collections;
using UnityEngine;
using Unity.Barracuda;

namespace MNISTPainter
{
    //must implement Dispose()
    public class Worker : IDisposable
    {
        //specifies 'public' objects
        private IWorker worker;
        private Model model;
        private float[] results;
        private PrecompiledComputeOps ops;

        //initialise public objects: loads MNIST model, creates a pool of workers to do inference,
        //initialises relevant GPU architecture
        public Worker(NNModel nnmodel, WorkerFactory.Type type)
        {
            bool verbose = false;
            model = ModelLoader.Load(nnmodel, verbose);
            worker = WorkerFactory.CreateWorker(type, model, verbose);

            var kernels = ComputeShaderSingleton.Instance.kernels;
            ops = new PrecompiledComputeOps(kernels, kernels[0]);
        }

        //method for cleaning garbage, adapted for gpu 
        public void Dispose()
        {
            worker?.Dispose();
            model = null;
            ops = null;
        }

        //runs through pixels in image, converts to model executable tensor, breaks each time a new column
        //is finished, and begins to execute prediction based on this updated information, 
        //outputs the model prediction
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
                yield return worker.StartManualSchedule(input);
            }

            Tensor output = worker.PeekOutput();

            results = output.data.Download(output.shape);

            input.Dispose();
            output.Dispose();
        }

        public IEnumerator ExecuteAsyncTexture(Texture2D texture)
        {
            Tensor input = new Tensor(texture, 1);
            yield return worker.StartManualSchedule(input);

            Tensor output = worker.PeekOutput();
            results = output.data.Download(output.shape);

            input.Dispose();
            output.Dispose();
        }

        public float[] get_result()
        {
            return results;
        }
    }
}
