// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package com.example.yolov8detect;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.yolov8detect.ml.WoodDetector;

//import org.pytorch.IValue;
//import org.pytorch.LiteModuleLoader;

import org.pytorch.Device;
import org.pytorch.Module;
//import com.google.flatbuffers.FlatBufferBuilder;

import org.pytorch.IValue;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicInteger;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

public class MainActivity extends AppCompatActivity implements Runnable {
    private int mImageIndex = 0;
    private final String[] mTestImages = {"test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg",};

    private String modelPath = null;
    private OrtSession session = null;
    private OrtEnvironment env = null;
    private TextView confidenceText;
    private TextView nmsLimitText;
    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;

    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    private final float max = 1.0f;
    private final float min = 0.0f;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);
        mResultView = findViewById(R.id.resultView);

        confidenceText = findViewById(R.id.confidenceText);
        nmsLimitText = findViewById(R.id.nmsLimitText);

        //private model = W.newInstance(getApplicationContext());
        SeekBar confidence = findViewById(R.id.seekBarConfidence);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            confidence.setMin(1);
        }
        confidence.setMax(100);
        confidence.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress,
                                          boolean fromUser) {
                float threshold = ((float) progress / (float) seekBar.getMax()) + min;
                PrePostProcessor.CONFIDENCE_THRESHOLD = threshold;
                confidenceText.setText(String.format("%s", threshold));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        confidence.setProgress(85);

        SeekBar nmsLimit = findViewById(R.id.seekBarNMSLimit);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nmsLimit.setMin(1);
        }
        nmsLimit.setMax(300);
        nmsLimit.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress,
                                          boolean fromUser) {
                PrePostProcessor.mNmsLimit = progress;
                nmsLimitText.setText(String.format("%s", progress));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        nmsLimit.setProgress(100);

        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(String.format("Test Image 1/%d", mTestImages.length));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.length));

                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });

        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(v -> {
            mResultView.setVisibility(View.INVISIBLE);

            final CharSequence[] options = { "Choose from Photos", "Take Picture", "Cancel" };
            AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
            builder.setTitle("New Test Image");

            builder.setItems(options, (dialog, item) -> {
                if (options[item].equals("Take Picture")) {
                    Intent takePicture = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(takePicture, 0);
                }
                else if (options[item].equals("Choose from Photos")) {
                    Intent pickPhoto = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                    startActivityForResult(pickPhoto , 1);
                }
                else if (options[item].equals("Cancel")) {
                    dialog.dismiss();
                }
            });
            builder.show();
        });

        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(v -> {
          final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
          startActivity(intent);
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(v -> {
            mButtonDetect.setEnabled(false);
            mProgressBar.setVisibility(ProgressBar.VISIBLE);
            mButtonDetect.setText(getString(R.string.run_model));

            mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.mInputWidth;
            mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.mInputHeight;

            mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
            mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());

            mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
            mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

            Thread thread = new Thread(MainActivity.this);
            thread.start();
        });

        try {
            mModule = Module.load(MainActivity.assetFilePath(getApplicationContext(), "yolov8-best-nano.torchscript"), null, Device.VULKAN);
//            modelPath = MainActivity.assetFilePath(getApplicationContext(), "yolov8-best-nano.with_runtime_opt.ort");
//
//            env = OrtEnvironment.getEnvironment();
//
//            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
//
//            sessionOptions.addConfigEntry("session.load_model_format", "ORT");
////          sessionOptions.addConfigEntry("session.execution_mode", "ORTGPU");
////          sessionOptions.addConfigEntry("session.gpu_device_id", "0");
//            sessionOptions.addConfigEntry("kOrtSessionOptionsConfigAllowIntraOpSpinning", "0");
//
////          Map<String, String> xnn = new HashMap<>();
////          xnn.put("intra_op_num_threads","1");
////          sessionOptions.addXnnpack(xnn);
//            sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
//            sessionOptions.setIntraOpNumThreads(2);
//              //sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL);
//              //sessionOptions.setCPUArenaAllocator(true);
//              //EnumSet<NNAPIFlags> flags = EnumSet.of(NNAPIFlags.USE_FP16);
//              sessionOptions.addNnapi();
////            //sessionOptions.addCPU(true);
//
//            Map<String, String> config = sessionOptions.getConfigEntries();
//
//            session = env.createSession(modelPath, sessionOptions);

            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);

        } catch (IOException | RuntimeException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
//        catch (OrtException e) {
//            throw new RuntimeException(e);
//        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }

    @Override
    public void run() {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);

        int width = resizedBitmap.getWidth();
        int height = resizedBitmap.getHeight();

        int size = Math.min(height, width);

        TensorImage img = TensorImage.fromBitmap(resizedBitmap);

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(size, size))
                        .add(new ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(127.5f, 127.5f)).build();

        img = imageProcessor.process(img);

        //float[] in = inputTensor.getDataAsFloatArray();
        float[] output = null;
        float[] output2 = null;

        try {

            // Initialize interpreter with GPU delegate
            Model.Options options;
            options = new Model.Options.Builder().setNumThreads(4).setDevice(Model.Device.CPU).build();

            WoodDetector model = WoodDetector.newInstance(getApplicationContext(), options);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3}, DataType.FLOAT32);

            inputFeature0.loadBuffer(img.getBuffer());

            // Runs model inference and gets result.

            WoodDetector.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();

            output = outputFeature0.getFloatArray();
            output2 = outputFeature1.getFloatArray();
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

        //ONNX-Runtime
//        Map<String, OnnxTensor> inputs = new HashMap<>();
//        try {
//            inputs.put("images", OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensor.getDataAsFloatArray()), inputTensor.shape()));
//        } catch (OrtException e) {
//            throw new RuntimeException(e);
//        }
//
//        float[] outOnnx = null
//        try (OrtSession.Result results = session.run(inputs)) {
//            Optional<OnnxValue> outputOnnx = results.get("output0");
//            if(outputOnnx.isPresent()){
//                final OnnxTensor t = OnnxTensor.createTensor(env, outputOnnx.get().getValue());
//                outOnnx = t.getFloatBuffer().array();
//            }
//        }
//        catch(Exception e){
//            throw new RuntimeException(e);
//        }

//        PYTORCH
//        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
//        PyTorchAndroid.setNumThreads(4);
//        final Tensor outputTensor = outputTuple[0].toTensor();
//        final float[] outputs = outputTensor.getDataAsFloatArray();


//        AtomicInteger max = new AtomicInteger();
//        AtomicInteger min = new AtomicInteger();
//        maximum.ifPresent(max::set);
//        minimum.ifPresent(min::set);

         //Denormalize according to https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1

        ArrayList<Result> resultList = new ArrayList<>();

        float scale = 127.5f; // 255

        for(int i = 0; i < 8400; i++){
            //output[i] = (output[i] - 0) * ((640 - 0) / 1) + 0;

            float cnf = output[i + 8400 * 4];
            if(cnf >= PrePostProcessor.CONFIDENCE_THRESHOLD) {

                float cx = output[i] * 640;

                float cy = output[i + 8400] * 640;

                float w = output[i + 8400 * 2] * 640;

                float h = output[i + 8400 * 3] * 640;

                float x1 = cx - (w/2F);
                float y1 = cy - (h/2F);
                float x2 = cx + (w/2F);
                float y2 = cy + (h/2F);

                Rect rect = new Rect((int)(mStartX+mIvScaleX*x1), (int)(mStartY+y1*mIvScaleY), (int)(mStartX+mIvScaleX*x2), (int)(mStartY+mIvScaleY*y2));
                resultList.add(new Result(0, cnf, rect));
                int a = 0;
            }
        }
//
        final ArrayList<Result> res = PrePostProcessor.nonMaxSuppression(resultList, PrePostProcessor.mNmsLimit, PrePostProcessor.CONFIDENCE_THRESHOLD);

        //final ArrayList<Result> results =  PrePostProcessor.outputsToNMSPredictions(outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);

        runOnUiThread(() -> {
            mButtonDetect.setEnabled(true);
            mButtonDetect.setText(getString(R.string.detect));
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(res);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
        });
    }
}