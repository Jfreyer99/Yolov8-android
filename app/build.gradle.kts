import com.android.build.api.dsl.AaptOptions
import com.android.build.api.dsl.AndroidResources

plugins {
    id("com.android.application")
}

android {
    namespace = "com.example.yolov8detect"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.yolov8detect"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

 androidResources

    buildFeatures {
        mlModelBinding = true
    }
}

dependencies {

    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    implementation ("androidx.camera:camera-core:1.0.0-alpha05")
    implementation ("androidx.camera:camera-camera2:1.0.0-alpha05")

    // https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android
    implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")

    implementation ("org.pytorch:pytorch_android_lite:1.12.2")
    implementation ("org.pytorch:pytorch_android_torchvision_lite:1.12.2")

    implementation("com.google.android.material:material:1.10.0")

    // https://mvnrepository.com/artifact/org.tensorflow/tensorflow-lite
    implementation("org.tensorflow:tensorflow-lite:2.14.0")


    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-metadata:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.3.0")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}