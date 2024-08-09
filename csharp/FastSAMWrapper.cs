using System;
using System.Collections.Generic;
using System.Linq;
using Python.Runtime;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace FastSAMWrapper
{
    public class FastSAMProcessor
    {
        public static (List<int>, List<Tuple<double, double>>) ProcessImage(
            string imgPath,
            float conf,
            string modelPath = @"C:\Users\HistoliX\Documents\FastSAM-needle-biopsy\weights\FastSAM-x.pt",
            int imgsz = 1024,
            float iou = 0.7F,
            string output = @"C:\Users\HistoliX\Documents\FastSAM-needle-biopsy\output",
            string pointPrompt = "[[0,0]]",
            string pointLabel = "[0]",
            string boxPrompt = "[[0,0,0,0]]",
            bool betterQuality = false,
            string device = null,
            bool retina = true,
            bool withContours = false,
            string microDims = "35,27",
            bool plot = true,
            int crop_pixels = 100 
        )
        {
            List<int> path = new List<int>();
            List<Tuple<double, double>> coordinates = new List<Tuple<double, double>>();

            try
            {
                // Initialize the Python runtime
                Runtime.PythonDLL = @"C:\Users\HistoliX\AppData\Local\Programs\Python\Python312\Python312.dll";
                PythonEngine.Initialize();

                using (Py.GIL())
                {
                    // Import necessary modules
                    dynamic sys = Py.Import("sys");
                    string scriptDirectory = @"C:\Users\HistoliX\Documents\FastSAM-needle-biopsy";
                    sys.path.append(scriptDirectory);

                    try
                    {
                        dynamic script = Py.Import("FastSAM_img_segmentation");
                        dynamic pythonResult = script.img_segment(
                            model_path: modelPath,
                            img_path: imgPath,
                            imgsz: imgsz,
                            iou: iou,
                            conf: conf,
                            output: output,
                            point_prompt: pointPrompt,
                            point_label: pointLabel,
                            box_prompt: boxPrompt,
                            better_quality: betterQuality,
                            device: device,
                            retina: retina,
                            withContours: withContours,
                            microDims: microDims,
                            plot: plot,
                            crop_pixels: crop_pixels
                        );

                        // Parse the JSON result
                        string jsonResult = pythonResult.ToString();
                        JObject parsedResult = JObject.Parse(jsonResult);

                        path = parsedResult["path"].ToObject<List<int>>();
                        List<List<double>> coordsList = parsedResult["coordinates"].ToObject<List<List<double>>>();

                        // Convert List<List<double>> to List<Tuple<double, double>>
                        coordinates = coordsList.Select(c => new Tuple<double, double>(c[0], c[1])).ToList();
                    }
                    catch (PythonException pyEx)
                    {
                        Console.Error.WriteLine($"Python error: {pyEx.Message}");
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"Error processing image: {ex.Message}");
                        Console.Error.WriteLine($"Stack trace: {ex.StackTrace}");
                    }
                }
            }
            finally
            {
                PythonEngine.Shutdown();
            }

            return (path, coordinates);
        }
    }
}