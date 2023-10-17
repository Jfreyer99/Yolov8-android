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

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.yolov8detect.ml.WoodDetector;

import org.pytorch.Module;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicInteger;

public class MainActivity extends AppCompatActivity implements Runnable {
    private int mImageIndex = 0;
    private final String[] mTestImages = {"test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg",};

    private String modelPath = null;
    //private OrtSession session = null;
    //private OrtEnvironment env = null;
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
            confidence.setMin(12);
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
              mModule = Module.load(MainActivity.assetFilePath(getApplicationContext(), "small.torchscript"));

//            modelPath = MainActivity.assetFilePath(getApplicationContext(), "small2.with_runtime_opt.ort");
//
//            env = OrtEnvironment.getEnvironment();
//
//            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
//
//            sessionOptions.addConfigEntry("session.load_model_format", "ORT");
//
//            //Map<String, String> xnn = new HashMap<>();
//            //xnn.put("intra_op_num_threads","1");
//
//            //sessionOptions.addXnnpack(xnn);
//            sessionOptions.addConfigEntry("kOrtSessionOptionsConfigAllowIntraOpSpinning", "0");
//            sessionOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
//
//            EnumSet<NNAPIFlags> flags = EnumSet.of(NNAPIFlags.USE_FP16);
//            sessionOptions.setIntraOpNumThreads(2);
//            sessionOptions.setInterOpNumThreads(2);
//            sessionOptions.addNnapi(flags);
//
//            //sessionOptions.addCPU(true);
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
        //final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);

//        Tensorflow
        TensorImage tensorImage = TensorImage.fromBitmap(resizedBitmap);

        int[] values = tensorImage.getTensorBuffer().getIntArray();

        OptionalInt minimum = Arrays.stream(values).min();
        OptionalInt maximum = Arrays.stream(values).max();

        minimum.ifPresent(System.out::println);
        maximum.ifPresent(System.out::println);

        float[] out = null;
        try {

            // Initialize interpreter with GPU delegate
            Model.Options options;
            CompatibilityList compatList = new CompatibilityList();
            options = new Model.Options.Builder().setNumThreads(4).build();
//            if(compatList.isDelegateSupportedOnThisDevice()){
//                // if the device has a supported GPU, add the GPU delegate
//                options = new Model.Options.Builder().setDevice(Model.Device.GPU).build();
//            } else {
//                // if the GPU is not supported, run on 4 threads
//                options = new Model.Options.Builder().setNumThreads(4).build();
//            }

            @androidx.annotation.NonNull WoodDetector model = WoodDetector.newInstance(getApplicationContext(), options);

            // Runs model inference and gets result.
            WoodDetector.Outputs outputs = model.process(tensorImage);

            WoodDetector.DetectionResult r = outputs.getDetectionResultList().get(0);

            TensorBuffer buf = outputs.getCategoryAsTensorBuffer();
            out = buf.getFloatArray();

            // Gets result from DetectionResult.
            //String location = detectionResult.getCategoryAsString();
            //RectF category = detectionResult.getLocationAsRectF();

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception

//        ONNX-Runtime
//        final Map<String, OnnxTensor> inputs;
//        try {
//            inputs = Map.of("images", OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensor.getDataAsFloatArray()), inputTensor.shape()));
//        } catch (OrtException e) {
//            throw new RuntimeException(e);
//        }
//
//        float[] out = null;
//        try (var results = session.run(inputs)) {
//            var output = results.get("output0");
//            if(output.isPresent()){
//                final OnnxTensor t = OnnxTensor.createTensor(env, output.get().getValue());
//                out = t.getFloatBuffer().array();
//            }
//        }
//        catch(Exception e){
//            throw new RuntimeException(e);
//        }

//        PYTORCH
//        IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
//        final Tensor outputTensor = outputTuple[0].toTensor();
//        final float[] outputs = outputTensor.getDataAsFloatArray();

//        for(int i = 0; i < out.length; i++){
//            out[i] = out[i]
//        }
        AtomicInteger max = new AtomicInteger();
        AtomicInteger min = new AtomicInteger();
        maximum.ifPresent(max::set);
        minimum.ifPresent(min::set);
        // Denormalize according to https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        for(int i = 0; i < out.length; i++){
            out[i] = out[i] + 1.0f * (( (float) max.get()- (float) min.get()) / 2) + min.get();
        }

        final ArrayList<Result> results =  PrePostProcessor.outputsToNMSPredictions(out, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);

        runOnUiThread(() -> {
            mButtonDetect.setEnabled(true);
            mButtonDetect.setText(getString(R.string.detect));
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(results);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
        });
    }
}