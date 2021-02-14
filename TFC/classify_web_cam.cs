using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;
using System.IO;
using System.Linq;

public class Webcam : MonoBehaviour
{
    //load the model onto the current thread
    public NNModel mob_net;
    public Model model;
    //load the concurrent process manager
    private IWorker worker;
    
    //pick the right system camera
    int current_cam_index = 0;
    
    //grab the webcam texture and load it onto the current thread
    WebCamTexture tex;
    
    //specify where we want to output the camera pixels onto on screen
    public RawImage display;

    public RawImage output;
   
    //UI feature for activating Neural network
    private bool brain_on = false;

    //constants required by MOBILE-NET architecture to do inference
    private const int SIZE = 224;
    private const int MEAN = 127;
    private const float STD_DEV = 127.5f;
    
    //initialise the the inference engine with our custom model
    void Start()
    {
        model = ModelLoader.Load(mob_net);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }
    
    //handle turning the camera on and off: extracting the whole image
    public void start_cam()
    {
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

    //event: user stops camera
    public void stop_cam()
    {
        display.texture = null;
        tex.Stop();
        tex = null;
    }
    
    void crop_normalize_inference(WebCamTexture src)
    {
        //location on screen of centre of image for analysis
        int x = Mathf.FloorToInt(display.transform.position.x);
        int y = Mathf.FloorToInt(display.transform.position.y);
        
        //1d array of pixels from the image, and crop them to correct size
        Color[] pix = src.GetPixels(x, y, SIZE, SIZE);
        
        //2d array of pixels
        Texture2D dest = new Texture2D(SIZE, SIZE);
        
        //for debugging
        dest.SetPixels(pix);
        dest.Apply();
        
        //normalize pixels
        float[] floats = new float[SIZE * SIZE * 3];

        for (int i = 0; i < pix.Length; ++i)
        {
            var color = pix[i];

            floats[i * 3 + 0] = (color.r - MEAN) / STD_DEV;
            floats[i * 3 + 1] = (color.g - MEAN) / STD_DEV;
            floats[i * 3 + 2] = (color.b - MEAN) / STD_DEV;
        }
        
        //load normalized pixels into array for inference
        Tensor in_tensor = new Tensor(0, SIZE, SIZE, 3, floats);
        
        //execute inference
        worker.Execute(in_tensor);
        
        //load output prediction onto local execution
        Tensor out_tensor = worker.PeekOutput();

        //find output predictions corresponding item index in data
        var max = Mathf.Max(out_tensor.ToReadOnlyArray());
        var arr = out_tensor.ToReadOnlyArray();
        var index = System.Array.IndexOf(arr, max);
        
        //load human readable prediction
        string line = File.ReadLines(@"D:\Unity\WebCam\Cam\Assets\Scenes\mobile_net.txt").Skip(index).Take(1).First();
        
        //print prediction
        Debug.Log(line);
        
        //memory management
        worker.Dispose();
        in_tensor.Dispose();
        out_tensor.Dispose();
    }
    
    //event: user shuts off inference engine
    public void brain()
    {
        brain_on = !brain_on;
    }
    
    //screen refresh handling
    private void Update()
    {
        if (brain_on)
        {
            crop_normalize_inference(tex);

            brain_on = false;
        }
    }
}
