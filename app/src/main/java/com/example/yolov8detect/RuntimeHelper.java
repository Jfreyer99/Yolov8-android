package com.example.yolov8detect;

import android.content.Context;
import android.graphics.Bitmap;

import com.example.yolov8detect.ml.WoodDetector;
import com.example.yolov8detect.ml.WoodDetectorFP16;
import com.example.yolov8detect.ml.WoodSSD;

import org.pytorch.Device;
import org.pytorch.IValue;


import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

public class RuntimeHelper {

    public enum RunTime{
        Onnx,
        PyTorch,
        TFLite,
        TFLITE_SSD
    }

    public static class SSDResult{
        public float[] scores;
        public float[] boxes;

        public SSDResult(float[] scores, float[] boxes) {
            this.scores = scores;
            this.boxes = boxes;
        }

        public float[] getScores() {
            return scores;
        }

        public float[] getBoxes() {
            return boxes;
        }
    }

    public static RunTime currentRuntime = RunTime.Onnx;

    public static OrtSession session = null;
    public static OrtEnvironment env = null;
    public static Module mModule = null;

    public static WoodDetector model;

    public static WoodDetectorFP16 modelFP16;

    public static WoodSSD woodSSD;

    private static float[] output;

    private static SSDResult ssdResult;

    public RuntimeHelper(){
        output = new float[0];
    }

    /**
     *
     * @param ctx Context of the Application
     * @param assetName Name of the asset inside the asset folder
     * @param backend choose the backend on which the inference runs on
     */
    public static void createOnnxRuntime(Context ctx, String assetName, String backend){
        try {

            String modelPath = MainActivity.assetFilePath(ctx, assetName);
            env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

            sessionOptions.addConfigEntry("session.load_model_format", "ORT");
//          sessionOptions.addConfigEntry("session.execution_mode", "ORTGPU");
//          sessionOptions.addConfigEntry("session.gpu_device_id", "0");
            sessionOptions.addConfigEntry("kOrtSessionOptionsConfigAllowIntraOpSpinning", "0");

//          Map<String, String> xnn = new HashMap<>();
//          xnn.put("intra_op_num_threads","1");
//          sessionOptions.addXnnpack(xnn);

            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            sessionOptions.setIntraOpNumThreads(4);
            sessionOptions.setInterOpNumThreads(4);
            sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);

            switch (backend.toUpperCase()){
                case "NNAPI":
                    EnumSet<NNAPIFlags> flags = EnumSet.of(NNAPIFlags.USE_FP16);
                    sessionOptions.addNnapi(flags);
                    break;
                case "CPU":
                    sessionOptions.addCPU(true);
                    break;
            }

            sessionOptions.setCPUArenaAllocator(true);
            session = env.createSession(modelPath, sessionOptions);

        } catch(IOException | OrtException e){

        }
    }

    /**
     *
     * @param inputTensor Tensor to run inference on
     * @return float[] for PostProcessing
     */
    public static Optional<float[]> invokeOnnxRuntime(Tensor inputTensor){
        //ONNX-Runtime
        float[] outOnnx = null;

        Map<String, OnnxTensor> inputs = new HashMap<>();
        try {
            inputs.put("images", OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensor.getDataAsFloatArray()), inputTensor.shape()));
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

        try (OrtSession.Result results = session.run(inputs)) {
            Optional<OnnxValue> outputOnnx = results.get("output0");
            if(outputOnnx.isPresent()){
                final OnnxTensor t = OnnxTensor.createTensor(env, outputOnnx.get().getValue());
                outOnnx = t.getFloatBuffer().array();
            }
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return Optional.of(outOnnx);
    }


    /**
     *
     * @param size size of the bitmap
     * @return new instance of ImageProcessor for Preprocessing the Bitmap before inference
     */
    private static ImageProcessor buildImageProcessor(int size, int targetHeight, int targetWidth){
                return new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(size, size))
                        .add(new ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(127.5f, 127.5f)).build();
    }

    /**
     *
     * @param ctx Context of the Application
     * @param device Model.Device.NNAPI || Model.Device.CPU || Model.Device.GPU ----- GPU does not work currently
     */
    public static void createTensorFlowLiteRuntime(Context ctx, Model.Device device){

        try {
            Model.Options options;
            options = new Model.Options.Builder().setDevice(device).setNumThreads(4).build();
            RuntimeHelper.model = WoodDetector.newInstance(ctx, options);

            //RuntimeHelper.modelFP16 = WoodDetectorFP16.newInstance(ctx, options);

            System.out.println("a");
        } catch(Exception e){
            System.out.println(e);
        }

    }

    public static void createTensorFlowLiteRuntineSSD(Context ctx, Model.Device device){
        try {
            Model.Options options;
            options = new Model.Options.Builder().setDevice(device).setNumThreads(8).build();

            RuntimeHelper.woodSSD = WoodSSD.newInstance(ctx, options);

        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    public static Optional<SSDResult> invokeTensorFlowLiteRuntimeSSD(Bitmap bmp)
    {
        int width = bmp.getWidth();
        int height = bmp.getHeight();
        int size = Math.min(height, width);

        TensorImage img = TensorImage.fromBitmap(bmp);

        img = buildImageProcessor(size, 320, 320).process(img);

        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 320, 320, 3}, DataType.FLOAT32);
        inputFeature0.loadBuffer(img.getBuffer());

//        ymin = int(max(1,(boxes[i][0] * imH)))
//        xmin = int(max(1,(boxes[i][1] * imW)))
//        ymax = int(min(imH,(boxes[i][2] * imH)))
//        xmax = int(min(imW,(boxes[i][3] * imW)))

        // Runs model inference and gets result.
        WoodSSD.Outputs outputs = RuntimeHelper.woodSSD.process(inputFeature0);

        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
        //TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
        //TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();


        float[] scores = outputFeature0.getFloatArray();
        float[] boxes = outputFeature1.getFloatArray();

        SSDResult result = new SSDResult(scores, boxes);

        return Optional.of(result);
    }

    /**
     *
     * @param bmp Bitmap to run detection on
     * @return Optional that contains the float[] for postprocessing
     */
    public static Optional<float[]> invokeTensorFlowLiteRuntime(Bitmap bmp){
        int width = bmp.getWidth();
        int height = bmp.getHeight();
        int size = Math.min(height, width);

        TensorImage img = TensorImage.fromBitmap(bmp);

        img = buildImageProcessor(size, 640, 640).process(img);

        float[] output;
        //float[] output2 = null;

        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3}, DataType.FLOAT32);

        inputFeature0.loadBuffer(img.getBuffer());

        //WoodDetectorFP16.Outputs outputs = RuntimeHelper.modelFP16.process(inputFeature0);

        // Runs model inference and gets result.
        WoodDetector.Outputs outputs = model.process(inputFeature0);

        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        //TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();

        output = outputFeature0.getFloatArray();
        //output2 = outputFeature1.getFloatArray();

        // Releases model resources if no longer used.
        //this.model.close();

        return Optional.of(output);
    }

    /**
     *
     * @param ctx Context of the Application
     * @param assetName Name of the model inside the asset folder
     * @param device Device.CPU or Device.VULKAN
     */
    public static void usePyTorch(Context ctx, String assetName, Device device){
        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(ctx.getApplicationContext(), assetName));
            //mModule = Module.load(MainActivity.assetFilePath(ctx, assetName), null, device);
        } catch (IOException e){

        }
    }

    public static Optional<float[]> invokePyTorch(Tensor inputTensor, int numThreads){
        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
        PyTorchAndroid.setNumThreads(numThreads);
        final Tensor outputTensor = outputTuple[0].toTensor();
        final float[] outputs = outputTensor.getDataAsFloatArray();
        return Optional.of(outputs);
    }

    public static void setOutputs(float[] output){
        RuntimeHelper.output = output;
    }

    public static float[] getOutput(){
        return output;
    }

    public static SSDResult getSsdResult() {
        return ssdResult;
    }

    public static void setSsdResult(SSDResult ssdResult) {
        RuntimeHelper.ssdResult = ssdResult;
    }
}