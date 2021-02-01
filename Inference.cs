using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;


namespace MNISTPainter
{
    public class Inference : MonoBehaviour
    {

        [SerializeField]
        public pCanvas paint_canvas;

        [SerializeField]
        private NNModel MNIST_model;

        [SerializeField]
        private WorkerFactory.Type m_worker_type = WorkerFactory.Type.ComputePrecompiled;

        [SerializeField]
        private Text[] m_results_array;

        private Worker m_barracuda_worker;

        private string[] m_Labels = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

        private bool m_is_busy = false;

        private void Start()
        {
            paint_canvas.on_update_texture += texture_listener;

            m_barracuda_worker = new Worker(MNIST_model, m_worker_type);
        }

        private void texture_listener(Texture2D texture)
        {
            if (!m_is_busy)
            {
                StartCoroutine(execute(texture));
            }
        }

        IEnumerator execute(Texture2D texture)
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
            }

            for (int i = 0; i < result.Length; i++)
            {
                int largest_element = -1;
                float largest_value = -1f;
                foreach (KeyValuePair<int, float> keyValuePair in results)
                {
                    if (keyValuePair.Value >= largest_value)
                    {
                        largest_value = keyValuePair.Value;
                        largest_element = keyValuePair.Key;
                    }
                }

                m_results_array[i].text = $"{m_Labels[largest_element]}: {largest_value}";
                results.Remove(largest_element);
            }

            m_is_busy = false;
        }
    }
}
