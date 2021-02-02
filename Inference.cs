using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;

namespace MNISTPainter
{
    public class Inference : MonoBehaviour
    {

        //Developer UI handling
        [SerializeField]
        public pCanvas paint_canvas;

        //Note: Inference with MNIST is buggy in unity: memory leaks seem to occur due to the structure
        //of the neural net
        [SerializeField]
        private NNModel MNIST_model;

        [SerializeField]
        private WorkerFactory.Type m_worker_type = WorkerFactory.Type.ComputePrecompiled;

        [SerializeField]
        private Text[] m_results_array;

        //main algorithm 'public' objects
        private Worker m_barracuda_worker;

        private string[] m_Labels = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

        private bool m_is_busy = false;

        //initialise observer pattern
        //instantiate worker for our model and backend execution type
        private void Start()
        {
            paint_canvas.on_update_texture += texture_listener;

            m_barracuda_worker = new Worker(MNIST_model, m_worker_type);
        }

        //runs worker on mouse down
        private void texture_listener(Texture2D texture)
        {
            if (!m_is_busy)
            {
                StartCoroutine(ExecuteAsync(texture));
            }
        }

        //concurrency: each frame updates the estimated model output by calling the Worder script
        //updates the UI to show predictions based on worker output
        IEnumerator ExecuteAsync(Texture2D texture)
        {
            if (m_is_busy)
            {
                yield break;
            }

            m_is_busy = true;

            yield return m_barracuda_worker.ExecuteAsyncTexture(texture);

            var result = m_barracuda_worker.get_result();

            Dictionary<int, float> results = new Dictionary<int, float>();

            for (int i = 0; i < result.Length; i++)
            {
                results[i] = result[i];

                foreach (KeyValuePair<int, float> keyValuePair in results)
                {

                    m_results_array[i].text = $"{m_Labels[keyValuePair.Key]}: {keyValuePair.Value}";

                    //here we will do UI things
                }
            }

            m_is_busy = false;
        }
    }
}
