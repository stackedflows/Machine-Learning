using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;

public class Webcam : MonoBehaviour
{

    public NNModel mob_net;
    public Model model;
    private IWorker worker;

    int current_cam_index = 0;

    WebCamTexture tex;

    public RawImage display;

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

    private void classify()
    {
        var channels = 3;

        var cam_pix = tex.GetPixels(
            (int)display.transform.position.x,
            (int)display.transform.position.y,
            (int)tex.width,
            (int)tex.height
            );

        Texture2D input = new Texture2D(224, 224);
        input.SetPixels(cam_pix);
        input.Apply();

        var in_tensor = new Tensor(input, channels);

        worker.Execute(in_tensor);

        var out_tensor = worker.PeekOutput("output"); 
        var max_val = Mathf.Max(out_tensor.ToReadOnlyArray());
        var arr = out_tensor.ToReadOnlyArray();
        var index = System.Array.IndexOf(arr, max_val);

        UnityEngine.Debug.Log("Output = " + out_tensor[0]);     
        UnityEngine.Debug.Log("Max prob = " + max_val);
        UnityEngine.Debug.Log("Index of max = " + index);
    }

    public void stop_cam()
    {
        display.texture = null;
        tex.Stop();
        tex = null;
    }
}
