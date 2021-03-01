using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;

public class ToArray : MonoBehaviour
{
    public string[] new_arr = new string[1001];

    public string parsed = "";

    void Start()
    {
        for (int i = 0; i < 1001; i++)
        {
            string line = File.ReadLines(@"Assets/Main/Scene/mobile_net.txt").Skip(i).Take(1).First();
            string guess = Regex.Replace(line.Split()[0], @"[^0-9a-zA-Z\ ]+", "");
            new_arr[i] = guess;
        }

        foreach (string s in new_arr)
        {
            parsed = parsed + $"\"{s}\",";
        }
    }
} 
