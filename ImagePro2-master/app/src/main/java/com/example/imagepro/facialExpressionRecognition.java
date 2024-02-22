package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class facialExpressionRecognition {
    private Interpreter interpreter;
    private int INPUT_SIZE;
    private int height=0;
    private int width=0;
    private GpuDelegate gpuDelegate=null;
    private CascadeClassifier cascadeClassifier;
    // Constructor for facialExpressionRecognition class
    facialExpressionRecognition(AssetManager assetManager, Context context,String modelPath,int inputSize)throws IOException {
        INPUT_SIZE=inputSize;
        // set gpu for interpreter
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        // add gpuDelegate to option
        options.addDelegate(gpuDelegate);
        //now set number of thread to options
        options.setNumThreads(4);
        interpreter = new Interpreter(loadModelFile(assetManager,modelPath),options);
        //if model is load right
        Log.d("facial_Expression","Model is loaded");
        //load haarcascade classifier
        try{
            //define input stream to read classifier
            InputStream is=context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            //creat a file and file in the file we create
            File cascadeDir=context.getDir("cascade",Context.MODE_PRIVATE);
            File mCascadeFile=new File(cascadeDir,"haarcascade_frontalface_alt");
            //define output stream to transfer data to file we create
            FileOutputStream os=new FileOutputStream(mCascadeFile);
            //create buffer to store byte
            byte[] buffer=new byte[4096];
            int byteRead;
            //read byte in while loop
            //when it read -1 = no data to read
            while ((byteRead=is.read(buffer))!=-1){
                //writing on mCascade file
                os.write(buffer,0,byteRead);
            }
            //close inout and output stream
            is.close();
            os.close();
            cascadeClassifier=new CascadeClassifier(mCascadeFile.getAbsolutePath());
            //if cascade file is load print
            Log.d("facial_Expression","Classifier is loaded");
        }catch (IOException e){
            e.printStackTrace();
        }

    }
    //load model
    public Mat recognizeImage(Mat mat_image){
        Core.flip(mat_image.t(),mat_image,1);//rotate image 90degree
        //convert mat_image to gray scale image
        Mat grayscaleImage = new Mat();
        Imgproc.cvtColor(mat_image,grayscaleImage,Imgproc.COLOR_RGBA2GRAY);
        //set height and width
        height=grayscaleImage.height();
        width=grayscaleImage.height();
        //define minimum height of face in original image
        int absoluteFaceSize = (int)(height*0.1);
        MatOfRect faces=new MatOfRect();
        //check if cascadeClassifier is load or not
        if(cascadeClassifier !=null){
            cascadeClassifier.detectMultiScale(grayscaleImage,faces,1.1,2,2
                    ,new Size(absoluteFaceSize,absoluteFaceSize),new Size());
        }
        //convert to array
        Rect[] faceArray=faces.toArray();
        //loop through each face
        for (int i =0;i<faceArray.length;i++){
            //draw rectangle around face
            Imgproc.rectangle(mat_image,faceArray[i].tl(),faceArray[i].br(),new Scalar(0,255,0,255),2);
            //crop face from original frame and gray scaleImage
            Rect roi=new Rect((int)faceArray[i].tl().x,(int)faceArray[i].tl().y,
                    ((int)faceArray[i].br().x)-(int)(faceArray[i].tl().x),
                    ((int)faceArray[i].br().y)-(int) (faceArray[i].tl().y));
            Mat cropped_rgba =new Mat(mat_image,roi);
            //now convert cropped_rgba to bitmap
            Bitmap bitmap=null;
            bitmap=Bitmap.createBitmap(cropped_rgba.cols(),cropped_rgba.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(cropped_rgba,bitmap);
            Bitmap scaleBitmap=Bitmap.createScaledBitmap(bitmap,48,48,false);
            //convert scaledBitmap to byteBuffer
            ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaleBitmap);
            //create object to hold output
            float[][] emotion=new float[1][1];
            //predict with bytebuffer as an input and emotion as an output
            interpreter.run(byteBuffer,emotion);
            //define float value of emotion;
            float emotion_v=(float)Array.get(Array.get(emotion,0),0);
            //if emotion is recognize print value
            Log.d("facial_expression","Output:"+emotion_v);
            //create function that return text emotion
            String emotion_s=get_emotion_text(emotion_v);
            //put text on original frame(mat_image)
            Imgproc.putText(mat_image,emotion_s+"("+emotion_v+")",
                    new Point((int)faceArray[i].tl().x+10,(int)faceArray[i].tl().y+20),
                    1,1.5,new Scalar(0,0,255,150),2);



        }
        Core.flip(mat_image.t(),mat_image,0);//rotate image -90 degree
        return mat_image;
    }
    // Method to convert emotion value to text
    private String get_emotion_text(float emotion_v) {
        String val="";
        //use if statement to determine val
        if(emotion_v>=0&emotion_v<0.5){
            val="Surprise";
        }
        else if(emotion_v>=0.5&emotion_v<1.5){
            val="Fear";
        }
        else if(emotion_v>=1.5&emotion_v<2.5){
            val="Angry";
        }
        else if(emotion_v>=2.5&emotion_v<3.5){
            val="Neutral";
        }
        else if(emotion_v>=3.5&emotion_v<4.5){
            val="Sad";
        }
        else if(emotion_v>=4.5&emotion_v<5.5){
            val="Disgust";
        }
        else {
            val = "Happy";
        }
        return val;
    }
     // Method to convert a bitmap to a ByteBuffer
    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaleBitmap) {
        ByteBuffer byteBuffer;
        int size_image=INPUT_SIZE;//48

        byteBuffer=ByteBuffer.allocateDirect(4*1*size_image*size_image*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_image*size_image];
        scaleBitmap.getPixels(intValues,0,scaleBitmap.getWidth(),0,0,scaleBitmap.getHeight(),scaleBitmap.getWidth());
        int pixel=0;
        for (int i =0;i<size_image;i++){
            for(int j=0;j<size_image;++j){
                final int val=intValues[pixel++];
                //put float value to bytebuffer
                //scale image to convert image from 0-255 to 0-1
                byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                byteBuffer.putFloat(((val&0xFF))/255.0f);
            }
        }
        return byteBuffer;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException{
        //give description of file
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelPath);
        //create a inputstream to read file
        FileInputStream inputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        // Access the file channel for efficient file reading
        FileChannel fileChannel = inputStream.getChannel();
        // Get the starting offset and declared length of the file
        long startOffset=assetFileDescriptor.getStartOffset();
        long declaredLength=assetFileDescriptor.getDeclaredLength();
        // Return the mapped byte buffer containing the model data
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
}
