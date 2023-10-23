// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package com.example.yolov8detect;

import android.graphics.Rect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

class Result {
    int classIndex;
    Float score;
    Rect rect;

    public Result(int cls, Float output, Rect rect) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
    }

    public float getScore(){
        return this.score;
    }
};

public class PrePostProcessor {
    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    // model input image size
    static int mInputWidth = 640;
    static int mInputHeight = 640;

    // model output is of size 25200*(num_of_class+5)
    private static int mOutputRow = 8400; // as decided by the YOLOv5 model for input image of size 640*640
    private static int mOutputColumn = 37; // left, top, right, bottom, score and 80 class probability
    private static float mThreshold = 0.78f; // score above which a detection is generated
    public static int mNmsLimit = 40;


    private static final int TENSOR_WIDTH = 640;
    private static final int TENSOR_HEIGHT = 640;
    private static final float INPUT_MEAN = 0f;
    private static final float INPUT_STANDARD_DEVIATION = 255f;

    private static final int NUM_ELEMENTS = 8400;
    private static final int NUM_CHANNELS = 37;
    private static final int BATCH_SIZE = 1;
    private static final int X_POINTS = 160;
    private static final int Y_POINTS = 160;
    private static final int MASKS_NUMBERS = 32;
    public static float CONFIDENCE_THRESHOLD = 0.78F;
    public static float IOU_THRESHOLD = 0.3F;

    static String[] mClasses;

    // The two methods nonMaxSuppression and IOU below are ported from https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
     Removes bounding boxes that overlap too much with other boxes that have
     a higher score.
     - Parameters:
     - boxes: an array of bounding boxes and their scores
     - limit: the maximum number of boxes that will be selected
     - threshold: used to decide whether boxes overlap too much
     */
    static ArrayList<Result> nonMaxSuppression(ArrayList<Result> boxes, int limit, float threshold) {

        boxes.sort(Comparator.comparing(Result::getScore));


        ArrayList<Result> selected = new ArrayList<>();
        int numBoxes = boxes.size();
        boolean[] active = new boolean[numBoxes];
        Arrays.fill(active, true);
        int numActive = numBoxes;

        for (int i = 0; i < numBoxes; i++) {
            if (numActive == 0 || selected.size() >= limit) {
                break;
            }
            if (active[i]) {
                Result boxA = boxes.get(i);
                selected.add(boxA);

                for (int j = i + 1; j < numBoxes; j++) {
                    if (active[j]) {
                        Result boxB = boxes.get(j);
                        if (IOU(boxA.rect, boxB.rect) > threshold) {
                            active[j] = false;
                            numActive--;

                            if (numActive <= 0) {
                                break;
                            }
                        }
                    }
                }
            }
        }
        return selected;
    }

    /**
     Computes intersection-over-union overlap between two bounding boxes.
     */
    static float IOU(Rect a, Rect b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        float areaB = (b.right - b.left) * (b.bottom - b.top);

        if (areaA <= 0 || areaB <= 0) {
            return 0.0f;
        }

        float intersectionMinX = (a.left > b.left) ? a.left : b.left;
        float intersectionMinY = (a.top > b.top) ? a.top : b.top;
        float intersectionMaxX = (a.right < b.right) ? a.right : b.right;
        float intersectionMaxY = (a.bottom < b.bottom) ? a.bottom : b.bottom;

        float intersectionWidth = (intersectionMaxX - intersectionMinX > 0) ? intersectionMaxX - intersectionMinX : 0;
        float intersectionHeight = (intersectionMaxY - intersectionMinY > 0) ? intersectionMaxY - intersectionMinY : 0;
        float intersectionArea = intersectionWidth * intersectionHeight;

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    static ArrayList<Result> outputsToNMSPredictions(float[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY) {

        ArrayList<Result> results = new ArrayList<>(100);

        for (int c = 0; c < NUM_ELEMENTS; c++) {
            float cnf = outputs[c + NUM_ELEMENTS * 4];
            if (cnf > CONFIDENCE_THRESHOLD) {
                float cx = outputs[c];
                float cy = outputs[c + NUM_ELEMENTS];
                float w = outputs[c + NUM_ELEMENTS * 2];
                float h = outputs[c + NUM_ELEMENTS * 3];

                float x1 = cx - (w/2F);
                float y1 = cy - (h/2F);
                float x2 = cx + (w/2F);
                float y2 = cy + (h/2F);

//                ArrayList<Float> maskWeight = new ArrayList<>();
//
//                for (int index = 0; index < MASKS_NUMBERS; index++) {
//                    maskWeight.add(outputs[c + NUM_ELEMENTS * (index + 5)]);
//                }

                Rect rect = new Rect((int)(startX+ivScaleX*x1), (int)(startY+y1*ivScaleY), (int)(startX+ivScaleX*x2), (int)(startY+ivScaleY*y2));
                results.add(new Result(0, cnf, rect));
                //results.add(Output0(cx = cx, cy = cy, w = w, h = h, cnf = cnf, maskWeight = maskWeight))
            }
        }

        return nonMaxSuppression(results, mNmsLimit, IOU_THRESHOLD);
    }

    static ArrayList<Result> outputsToNMSPredictionsTFLITE(float[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY){
        ArrayList<Result> resultList = new ArrayList<>();

        for(int i = 0; i < NUM_ELEMENTS; i++){

            float cnf = outputs[i + 8400 * 4];
            if(cnf >= CONFIDENCE_THRESHOLD) {

                float cx = outputs[i] * 640;
                float cy = outputs[i + 8400] * 640;
                float w = outputs[i + 8400 * 2] * 640;
                float h = outputs[i + 8400 * 3] * 640;

                float x1 = cx - (w/2F);
                float y1 = cy - (h/2F);
                float x2 = cx + (w/2F);
                float y2 = cy + (h/2F);

                Rect rect = new Rect((int)(startX+ivScaleX*x1), (int)(startY+y1*ivScaleY), (int)(startX+ivScaleX*x2), (int)(startY+ivScaleY*y2));
                resultList.add(new Result(0, cnf, rect));
            }
        }
        return nonMaxSuppression(resultList, mNmsLimit, IOU_THRESHOLD);
    }
}