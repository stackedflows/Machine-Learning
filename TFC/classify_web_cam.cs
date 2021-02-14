using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using System.IO;
using System.Linq;

public class Webcam : MonoBehaviour
{
    public NNModel mob_net;
    public Model model;
    private IWorker worker;

    int current_cam_index = 0;

    WebCamTexture tex;

    public RawImage display;

    public RawImage output;

    private bool brain_on = false;

    private const int SIZE = 224;
    private const int MEAN = 127;
    private const float STD_DEV = 127.5f;

    public void start_cam()
    {

        model = ModelLoader.Load(mob_net);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        if (tex != null)
        {
            stop_cam();
        }
        else
        {
            WebCamDevice device = WebCamTexture.devices[current_cam_index];
            tex = new WebCamTexture(device.name);
            display.texture = tex;
            tex.Play();
        }
    }

    public void stop_cam()
    {
        display.texture = null;
        tex.Stop();
        tex = null;
    }

 
    void crop_normalize_inference(WebCamTexture src)
    {
        int x = Mathf.FloorToInt(display.transform.position.x);
        int y = Mathf.FloorToInt(display.transform.position.y);

        Color[] pix = src.GetPixels(x, y, SIZE, SIZE);

        Texture2D dest = new Texture2D(SIZE, SIZE);

        dest.SetPixels(pix);
        dest.Apply();

        float[] floats = new float[SIZE * SIZE * 3];

        for (int i = 0; i < pix.Length; ++i)
        {
            var color = pix[i];

            floats[i * 3 + 0] = (color.r - MEAN) / STD_DEV;
            floats[i * 3 + 1] = (color.g - MEAN) / STD_DEV;
            floats[i * 3 + 2] = (color.b - MEAN) / STD_DEV;
        }

        Tensor in_tensor = new Tensor(0, SIZE, SIZE, 3, floats);

        worker.Execute(in_tensor);

        Tensor out_tensor = worker.PeekOutput();

        var max = Mathf.Max(out_tensor.ToReadOnlyArray());
        var arr = out_tensor.ToReadOnlyArray();
        var index = System.Array.IndexOf(arr, max);

        string line = File.ReadLines(@"D:\Unity\WebCam\Cam\Assets\Scenes\mobile_net.txt").Skip(index).Take(1).First();

        Debug.Log(line);
    }

    public void brain()
    {
        brain_on = !brain_on;
    }

    private void Update()
    {
        if (brain_on)
        {
            crop_normalize_inference(tex);

            brain_on = false;
        }
    }
}
