package com.example.yolov8detect;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;

import com.example.yolov8detect.ml.PytorchnDetect320Float32;
import com.example.yolov8detect.ml.WoodDetectorFP16;
import com.example.yolov8detect.ml.WoodSSD;
import com.example.yolov8detect.ml.WoodSSD640;

import org.pytorch.Device;
import org.pytorch.IValue;


import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TensorFlowLite;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import ai.onnxruntime.OnnxJavaType;
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
        TFLITE_SSD,
        TFLITE_SSD640
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

    //public static WoodDetector model;

    public static WoodDetectorFP16 modelFP16;

    public static WoodSSD640 woodSSD640;

    public static WoodSSD woodSSD;

    public static PytorchnDetect320Float32 model;

    public static Interpreter interpreter;


    private static float[] output;

    private static SSDResult ssdResult;

    public static int benchmarkSize = 50;
    public static long[] inference = new long[benchmarkSize];
    public static int counter = 0;

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

        Map<String, OnnxTensor> inputs = new HashMap<>();
        try {
            inputs.put("images", OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensor.getDataAsFloatArray()), inputTensor.shape()));
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

        try{

            long startTime = System.currentTimeMillis();
            OrtSession.Result results = session.run(inputs);
            long endTime = System.currentTimeMillis();
//            inference[counter] = endTime-startTime;
//            counter++;

            Optional<OnnxValue> outputOnnx = results.get("output0");
            if(outputOnnx.isPresent()){
                return Optional.of(OnnxTensor.createTensor(env, outputOnnx.get().getValue()).getFloatBuffer().array());
            }
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return Optional.of(null);
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
            RuntimeHelper.model = PytorchnDetect320Float32.newInstance(ctx, options);

            //RuntimeHelper.modelFP16 = WoodDetectorFP16.newInstance(ctx, options);

            System.out.println("a");
        } catch(Exception e){
            System.out.println(e);
        }

    }

    public static void createTensorFlowLiteRuntineSSD(Context ctx, Model.Device device, int modelSize){
        try {
            Model.Options options;
            options = new Model.Options.Builder().setDevice(device).setNumThreads(8).build();

            switch(modelSize){
                case 640: RuntimeHelper.woodSSD640 = WoodSSD640.newInstance(ctx, options);
                    break;
                case 320: RuntimeHelper.woodSSD = WoodSSD.newInstance(ctx, options);
                    break;
            }

        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    private static Optional<SSDResult> processSSD(TensorImage img, int size, int modelInputSize){
        img = buildImageProcessor(size, modelInputSize, modelInputSize).process(img);
        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, modelInputSize, modelInputSize, 3}, DataType.FLOAT32);
        inputFeature0.loadBuffer(img.getBuffer());
        // Runs model inference and gets result.

        if(modelInputSize == 640){

            long startTime = System.currentTimeMillis();
            WoodSSD640.Outputs outputs= RuntimeHelper.woodSSD640.process(inputFeature0);
            long endTime = System.currentTimeMillis();
//            inference[counter] = endTime-startTime;
//            counter++;

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
            //TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();// TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();
            float[] scores = outputFeature0.getFloatArray();
            float[] boxes = outputFeature1.getFloatArray();
            SSDResult result = new SSDResult(scores, boxes);
            return Optional.of(result);
        }

        long startTime = System.currentTimeMillis();
        WoodSSD.Outputs outputs= RuntimeHelper.woodSSD.process(inputFeature0);
        long endTime = System.currentTimeMillis();
//        inference[counter] = endTime-startTime;
//        counter++;

        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
        //TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();// TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();
        float[] scores = outputFeature0.getFloatArray();
        float[] boxes = outputFeature1.getFloatArray();

        SSDResult result = new SSDResult(scores, boxes);

        return Optional.of(result);
    }

    public static Optional<SSDResult> invokeTensorFlowLiteRuntimeSSD(Bitmap bmp, int modelInputSize)
    {
        int width = bmp.getWidth();
        int height = bmp.getHeight();
        int size = Math.min(height, width);

        TensorImage img = TensorImage.fromBitmap(bmp);

        return RuntimeHelper.processSSD(img, size, modelInputSize);
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

        img = buildImageProcessor(size, 320, 320).process(img);

        float[] output;
        //float[] output2 = null;

        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 320, 320, 3}, DataType.FLOAT32);

        inputFeature0.loadBuffer(img.getBuffer());

        //WoodDetectorFP16.Outputs outputs = RuntimeHelper.modelFP16.process(inputFeature0);

        // Runs model inference and gets result.
        //long startTime = System.currentTimeMillis();

        long startTime = System.currentTimeMillis();
        PytorchnDetect320Float32.Outputs outputs = model.process(inputFeature0);
        long endTime = System.currentTimeMillis();
//        inference[counter] = endTime-startTime;
//        counter++;

        //long endTime = System.currentTimeMillis();
        //System.out.println("Inference Time Tensorflow in ms " + (endTime-startTime));

        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        //TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();

        output = outputFeature0.getFloatArray();
        //output2 = outputFeature1.getFloatArray();

        // Releases model resources if no longer used.
        //this.model.close();

        return Optional.of(output);
    }


    public static Optional<float[]> invokeTensorFlowLiteRuntimeInterpreter(Context ctx, Bitmap bmp, String assetName) {

        File model = null;
        try {
            model = new File(MainActivity.assetFilePath(ctx.getApplicationContext(), assetName));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try(Interpreter interpreter = new Interpreter(model)) {
            int width = bmp.getWidth();
            int height = bmp.getHeight();
            int size = Math.min(height, width);

            TensorImage img = TensorImage.fromBitmap(bmp);

            img = buildImageProcessor(size, 320, 320).process(img);

            TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, 5, 2100}, DataType.FLOAT32);

            interpreter.run(img.getBuffer(), output.getBuffer());

            return Optional.of(output.getFloatArray());
        }
    }

    /**
     *
     * @param ctx Context of the Application
     * @param assetName Name of the model inside the asset folder
     */
    public static void usePyTorch(Context ctx, String assetName, int numThreads){
        try {
            PyTorchAndroid.setNumThreads(numThreads);
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(ctx.getApplicationContext(), assetName), null, Device.CPU);
        } catch (IOException e){

        }
    }

    public static Optional<float[]> invokePyTorchDetect(Tensor inputTensor){

        IValue input = IValue.from(inputTensor);

        long startTime = System.currentTimeMillis();
        IValue out = mModule.forward(input);
        long endTime = System.currentTimeMillis();

//        inference[counter] = endTime-startTime;
//        counter++;

        return Optional.of(out.toTensor().getDataAsFloatArray());
    }

    public static Optional<float[]> invokePyTorchSegment(Tensor inputTensor){
        IValue input = IValue.from(inputTensor);

        long startTime = System.currentTimeMillis();
        IValue out = mModule.forward(input);
        long endTime = System.currentTimeMillis();

//        inference[counter] = endTime-startTime;
//        counter++;

        IValue out0 = out.toTuple()[0];

        return Optional.of(out0.toTensor().getDataAsFloatArray());
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