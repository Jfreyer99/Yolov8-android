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
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.pytorch.Device;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.tensorflow.lite.support.model.Model;

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
import java.util.Optional;
import java.util.OptionalDouble;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;

public class MainActivity extends AppCompatActivity implements Runnable, AdapterView.OnItemSelectedListener {
    private int mImageIndex = 0;
    private final String[] mTestImages = {"test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg",};
    private TextView confidenceText;
    private TextView nmsLimitText;
    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;
    private final float max = 1.0f;
    private final float min = 0.0f;

    private long[] completeTime = new long[RuntimeHelper.benchmarkSize];
    private int counter = 0;

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

        Spinner spinner = (Spinner) findViewById(R.id.runtimeSelector);
        // Create an ArrayAdapter using the string array and a default spinner layout.
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(
                this,
                R.array.runtime_array,
                android.R.layout.simple_spinner_item
        );
        // Specify the layout to use when the list of choices appears.
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        // Apply the adapter to the spinner.
        spinner.setAdapter(adapter);
        spinner.setOnItemSelectedListener(this);


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

        ArrayList<Result> results = new ArrayList<>();
        switch(RuntimeHelper.currentRuntime){
            case Onnx:
                long startTime = System.currentTimeMillis();
                Bitmap resizedBitmapOnnx = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
                Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmapOnnx, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
                RuntimeHelper.invokeOnnxRuntime(inputTensor).ifPresent(RuntimeHelper::setOutputs);
                results =  PrePostProcessor.outputsToNMSPredictions(RuntimeHelper.getOutput(), mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
                long endTime = System.currentTimeMillis();
                //completeTime[counter] = endTime-startTime;
                break;
            case PyTorch:
                long startTime2 = System.currentTimeMillis();
                Bitmap resizedBitmapPyTorch = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
                Tensor inputTensorPy = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmapPyTorch, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
                RuntimeHelper.invokePyTorchDetect(inputTensorPy).ifPresent(RuntimeHelper::setOutputs);
                results =  PrePostProcessor.outputsToNMSPredictions(RuntimeHelper.getOutput(), mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
                long endTime2 = System.currentTimeMillis();
                //completeTime[counter] = endTime2-startTime2;
                break;
            case TFLite:
                  Bitmap resizedBitmapTFLite = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
                  RuntimeHelper.invokeTensorFlowLiteRuntimeInterpreter(getApplicationContext(), resizedBitmapTFLite, "pytorchn-detect-640_float32.tflite").ifPresent(RuntimeHelper::setOutputs);
                  results = PrePostProcessor.outputsToNMSPredictionsTFLITE(RuntimeHelper.getOutput(), mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
//                long startTime3 = System.currentTimeMillis();
//                Bitmap resizedBitmapTFLite = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
//                RuntimeHelper.invokeTensorFlowLiteRuntime(resizedBitmapTFLite).ifPresent(RuntimeHelper::setOutputs);
//                results = PrePostProcessor.outputsToNMSPredictionsTFLITE(RuntimeHelper.getOutput(), mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
//                long endTime3 = System.currentTimeMillis();
                //completeTime[counter] = endTime3-startTime3;
                break;
            case TFLITE_SSD:
                long startTime4 = System.currentTimeMillis();
                Bitmap resizedBitmapSSD = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
                RuntimeHelper.invokeTensorFlowLiteRuntimeSSD(resizedBitmapSSD, 320).ifPresent(RuntimeHelper::setSsdResult);
                results = PrePostProcessor.outputsTFLITESSD(RuntimeHelper.getSsdResult().scores, RuntimeHelper.getSsdResult().boxes, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
                long endTime4 = System.currentTimeMillis();
                //completeTime[counter] = endTime4-startTime4;
                break;
            case TFLITE_SSD640:
                long startTime5 = System.currentTimeMillis();
                Bitmap resizedBitmapSSD640 = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
                RuntimeHelper.invokeTensorFlowLiteRuntimeSSD(resizedBitmapSSD640, 640).ifPresent(RuntimeHelper::setSsdResult);
                results = PrePostProcessor.outputsTFLITESSD(RuntimeHelper.getSsdResult().scores, RuntimeHelper.getSsdResult().boxes, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
                long endTime5 = System.currentTimeMillis();
                //completeTime[counter] = endTime5-startTime5;
                break;
        }

//        if(counter == completeTime.length-1){
//            double meanOverall = calculateMeanOverall(completeTime);
//            double meanInference = calculateMeanInference(RuntimeHelper.inference);
//            System.out.println("Complete: " + meanOverall);
//            System.out.println("Inference " + meanInference);
//            System.out.println("Processing " + (meanOverall-meanInference));
//            RuntimeHelper.counter = 0;
//            counter = 0;
//        }

        final ArrayList<Result> finalResults = results;
        runOnUiThread(() -> {
            mButtonDetect.setEnabled(true);
            mButtonDetect.setText(getString(R.string.detect));
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(finalResults);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
        });
        //counter++;
    }

    public double calculateMeanOverall(long[] inputs){
        OptionalDouble avg = Arrays.stream(inputs).average();
        return avg.isPresent() ? avg.getAsDouble(): 0.0;
    }

    public double calculateMeanInference(long[] inputs){
        OptionalDouble avg = Arrays.stream(inputs).average();
        return avg.isPresent() ? avg.getAsDouble(): 0.0;
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

        switch (position){
            case 0:
                RuntimeHelper.createOnnxRuntime(getApplicationContext(), "pytorchn-detect-320.with_runtime_opt.ort", "NNAPI");
                RuntimeHelper.currentRuntime = RuntimeHelper.RunTime.Onnx;
                break;
            case 1: RuntimeHelper.usePyTorch(getApplicationContext(),"pytorchn-detect-320.torchscript", 4);
                RuntimeHelper.currentRuntime = RuntimeHelper.RunTime.PyTorch;
                break;
            case 2:
                RuntimeHelper.createTensorFlowLiteRuntime(getApplicationContext(), Model.Device.NNAPI);
                RuntimeHelper.currentRuntime = RuntimeHelper.RunTime.TFLite;
                break;
            case 3:
                RuntimeHelper.createTensorFlowLiteRuntineSSD(getApplicationContext(), Model.Device.CPU, 320);
                RuntimeHelper.currentRuntime = RuntimeHelper.RunTime.TFLITE_SSD;
                break;
            case 4:
                RuntimeHelper.createTensorFlowLiteRuntineSSD(getApplicationContext(), Model.Device.CPU, 640);
                RuntimeHelper.currentRuntime = RuntimeHelper.RunTime.TFLITE_SSD640;
                break;
        }
    }


    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        RuntimeHelper.createOnnxRuntime(getApplicationContext(), "yolov8-best-nano.with_runtime_opt.ort", "CPU");
        RuntimeHelper.currentRuntime = RuntimeHelper.RunTime.Onnx;
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(RuntimeHelper.modelFP16 != null){
            RuntimeHelper.modelFP16.close();
        }
        if(RuntimeHelper.mModule != null)
        {
            RuntimeHelper.mModule.destroy();
        }
        try {
            if(RuntimeHelper.session != null) {
                RuntimeHelper.session.close();
                RuntimeHelper.env.close();
            }
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        System.out.println("Destory");
    }
}