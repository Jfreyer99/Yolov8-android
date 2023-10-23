package com.example.yolov8detect;

import android.content.Context;
import android.graphics.Bitmap;

import com.example.yolov8detect.ml.WoodDetector;

import org.pytorch.Device;
import org.pytorch.IValue;
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

    private OrtSession session = null;
    private OrtEnvironment env = null;
    private Bitmap mBitmap = null;
    private Module mModule = null;

    private WoodDetector model;

    private float[] output;

    public RuntimeHelper(){
        output = new float[0];
    }

    /**
     *
     * @param ctx Context of the Application
     * @param assetName Name of the asset inside the asset folder
     * @param backend choose the backend on which the inference runs on
     */
    public void createOnnxRuntime(Context ctx, String assetName, String backend){
        //yolov8-best-nano.with_runtime_opt.ort
        try {

            String modelPath = MainActivity.assetFilePath(ctx, assetName);
            this.env = OrtEnvironment.getEnvironment();
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
            this.session = env.createSession(modelPath, sessionOptions);

        } catch(IOException | OrtException e){

        }
    }

    /**
     *
     * @param inputTensor Tensor to run inference on
     * @return float[] for PostProcessing
     */
    public Optional<float[]> invokeOnnxRuntime(Tensor inputTensor){
        //ONNX-Runtime
        float[] outOnnx = null;

        Map<String, OnnxTensor> inputs = new HashMap<>();
        try {
            inputs.put("images", OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensor.getDataAsFloatArray()), inputTensor.shape()));
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

        try (OrtSession.Result results = this.session.run(inputs)) {
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
    private ImageProcessor buildImageProcessor(int size){
                return new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(size, size))
                        .add(new ResizeOp(PrePostProcessor.mInputHeight, PrePostProcessor.mInputWidth, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(127.5f, 127.5f)).build();
    }

    /**
     *
     * @param ctx Context of the Application
     * @param device Model.Device.NNAPI || Model.Device.CPU || Model.Device.GPU ----- GPU does not work currently
     */
    public void createTensorFlowLiteRuntime(Context ctx, Model.Device device){

        try {
            Model.Options options;
            options = new Model.Options.Builder().setDevice(Model.Device.NNAPI).build();
            this.model = WoodDetector.newInstance(ctx, options);
        } catch(Exception e){

        }

    }

    /**
     *
     * @param bmp Bitmap to run detection on
     * @return Optional that contains the float[] for postprocessing
     */
    public Optional<float[]> invokeTensorFlowLiteRuntime(Bitmap bmp){
        int width = bmp.getWidth();
        int height = bmp.getHeight();
        int size = Math.min(height, width);

        TensorImage img = TensorImage.fromBitmap(bmp);

        img = buildImageProcessor(size).process(img);

        float[] output;
        float[] output2 = null;

        // Creates inputs for reference.
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3}, DataType.FLOAT32);

        inputFeature0.loadBuffer(img.getBuffer());

        // Runs model inference and gets result.
        WoodDetector.Outputs outputs = this.model.process(inputFeature0);

        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();

        output = outputFeature0.getFloatArray();
        output2 = outputFeature1.getFloatArray();

        // Releases model resources if no longer used.
        this.model.close();

        return Optional.of(output);
    }

    /**
     *
     * @param ctx Context of the Application
     * @param assetName Name of the model inside the asset folder
     * @param device Device.CPU or Device.VULKAN
     */
    public void usePyTorch(Context ctx, String assetName, Device device){
        try {
            this.mModule = Module.load(MainActivity.assetFilePath(ctx, assetName), null, device);
        } catch (IOException e){

        }
    }

    public Optional<float[]> invokePyTorch(Tensor inputTensor, int numThreads){
        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
        PyTorchAndroid.setNumThreads(numThreads);
        final Tensor outputTensor = outputTuple[0].toTensor();
        final float[] outputs = outputTensor.getDataAsFloatArray();
        return Optional.of(outputs);
    }

    public void setOutputs(float[] output){
        this.output = output;
    }

    public float[] getOutput(){
        return this.output;
    }

}
