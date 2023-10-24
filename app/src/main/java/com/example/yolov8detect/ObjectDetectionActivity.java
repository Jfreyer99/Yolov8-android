package com.example.yolov8detect;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.Device;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.support.model.Model;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Objects;

public class ObjectDetectionActivity extends AbstractCameraXActivity<ObjectDetectionActivity.AnalysisResult> {
    private Module mModule = null;
    private ResultView mResultView;

    static class AnalysisResult {
        private final ArrayList<Result> mResults;

        public AnalysisResult(ArrayList<Result> results) {
            mResults = results;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_object_detection;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        return ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mResultView.setResults(result.mResults);
        mResultView.invalidate();
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {

        if (RuntimeHelper.model == null) {
            RuntimeHelper.createTensorFlowLiteRuntime(getApplicationContext(), Model.Device.NNAPI);
        }

        Bitmap bitmap = imgToBitmap(Objects.requireNonNull(image.getImage()));
        Matrix matrix = new Matrix();
        matrix.postRotate(90.0f);

        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);

        float imgScaleX = (float)bitmap.getWidth() / PrePostProcessor.mInputWidth;
        float imgScaleY = (float)bitmap.getHeight() / PrePostProcessor.mInputHeight;
        float ivScaleX = (float) mResultView.getWidth() / bitmap.getWidth();
        float ivScaleY = (float)mResultView.getHeight() / bitmap.getHeight();

        ArrayList<Result> results = new ArrayList<>();

        switch(RuntimeHelper.currentRuntime){
            case Onnx:
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
                RuntimeHelper.invokeOnnxRuntime(inputTensor).ifPresent(RuntimeHelper::setOutputs);
                results =  PrePostProcessor.outputsToNMSPredictions(RuntimeHelper.getOutput(), imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
                break;
            case PyTorch:
                final Tensor inputTensorOnnx = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
                RuntimeHelper.invokePyTorch(inputTensorOnnx, 4).ifPresent(RuntimeHelper::setOutputs);
                results =  PrePostProcessor.outputsToNMSPredictions(RuntimeHelper.getOutput(), imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
                break;
            case TFLite:
                RuntimeHelper.invokeTensorFlowLiteRuntime(resizedBitmap).ifPresent(RuntimeHelper::setOutputs);
                results = PrePostProcessor.outputsToNMSPredictionsTFLITE(RuntimeHelper.getOutput(), imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
                break;
        }

        return new AnalysisResult(results);
    }
}