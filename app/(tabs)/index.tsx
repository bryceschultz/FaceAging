import React, { useState } from 'react';
import { StyleSheet, TouchableOpacity, Image, KeyboardAvoidingView, Platform, SafeAreaView, ScrollView, ActivityIndicator } from 'react-native';
import { Text, View } from '@/components/Themed';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { launchCamera } from 'react-native-image-picker';
import ImageResizer from 'react-native-image-resizer';
import RNFS from 'react-native-fs';
import { Buffer } from 'buffer';
import jpeg from 'jpeg-js';
import { convertToRGB } from 'react-native-image-to-rgb';

// Ensure Buffer is available globally
global.Buffer = Buffer;

export default function TabOneScreen() {
  const objectDetection = useTensorflowModel(require('../../assets/models/aging_gan_generator.tflite'));
  const model = objectDetection.state === 'loaded' ? objectDetection.model : undefined;
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [resizedPhotoUri, setResizedPhotoUri] = useState<string | null>(null);
  const [outputUri, setOutputUri] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const takePicture = async () => {
    const result = await launchCamera({
      mediaType: 'photo',
      includeBase64: true,
      maxHeight: 1024,
      maxWidth: 1024,
    });

    if (result.assets && result.assets.length > 0) {
      const photo = result.assets[0];
      if (photo.uri) {
        setPhotoUri(photo.uri);
        processImage(photo.uri);
      }
    }
  };

  const processImage = async (uri: string) => {
    if (model) {
      setOutputUri(null)
      setPhotoUri(null)
      setResizedPhotoUri(null)
      setLoading(true);
      try {
        console.log('Starting image processing...');
        const resizedUri = await resizeAndCropImage(uri, 512, 512);
        const inputTensor = await preprocessImage(resizedUri);
        console.log('Input Tensor Dimensions:', [1, 3, 512, 512]);
        const outputDict = model.runSync([inputTensor]);

        const outputTensor = outputDict[0];
        const reshapedOutput = reshapeOutput(new Float32Array(outputTensor));

        const transformedOutput = transformOutput(reshapedOutput);

        const outputUri = await postprocessOutput(transformedOutput);
        setOutputUri(outputUri);
        console.log('Image processing completed.');
      } catch (error) {
        console.error('Error processing image:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  const resizeAndCropImage = async (uri: string, targetWidth: number, targetHeight: number) => {
    const resizedImage = await ImageResizer.createResizedImage(uri, targetWidth, targetHeight, 'JPEG', 100, 0, undefined, false, {
      mode: 'cover',
    });
    const { uri: resizedUri } = resizedImage;
  
    // Load the resized image
    const imageData = await RNFS.readFile(resizedUri, 'base64');
    const buffer = Buffer.from(imageData, 'base64');
    const rawImageData = jpeg.decode(buffer, { useTArray: true });
  
    const { width, height, data } = rawImageData;
  
    if (width !== targetWidth) {
      throw new Error('Width of resized image does not match target width.');
    }
  
    let startY = 0;
    let endY = height;
  
    if (height > targetHeight) {
      const excessHeight = height - targetHeight;
      startY = excessHeight / 2;
      endY = startY + targetHeight;
    }
  
    const croppedData = [];
    for (let y = startY; y < endY; y++) {
      for (let x = 0; x < width; x++) {
        const index = (Math.floor(y) * width + x) * 4;
        croppedData.push(data[index], data[index + 1], data[index + 2], data[index + 3]);
      }
    }
  
    const frameData = {
      data: new Uint8Array(croppedData),
      width: targetWidth,
      height: targetHeight,
    };
  
    const croppedImageData = jpeg.encode(frameData, 100);
    const croppedPath = `${RNFS.DocumentDirectoryPath}/cropped.jpg`;
    await RNFS.writeFile(croppedPath, croppedImageData.data.toString('base64'), 'base64');
  
    setResizedPhotoUri(croppedPath);
    return croppedPath;
  };

  const preprocessImage = async (uri: string) => {
    const imageData = await RNFS.readFile(uri, 'base64');
    const buffer = Buffer.from(imageData, 'base64');
    const rawImageData = jpeg.decode(buffer, { useTArray: true });

    const { width, height, data } = rawImageData;

    if (width !== 512 || height !== 512) {
      throw new Error('Image dimensions must be 512x512');
    }

    const convertedArray = await convertToRGB(uri);

    const float32Array = new Float32Array(512 * 512 * 3);
    for (let i = 0; i < convertedArray.length; i++) {
      float32Array[i] = (convertedArray[i] / 255 - 0.5) / 0.5;
    }

    const reshapedArray = new Float32Array(1 * 3 * 512 * 512);
    for (let c = 0; c < 3; c++) {
      for (let h = 0; h < 512; h++) {
        for (let w = 0; w < 512; w++) {
          reshapedArray[c * 512 * 512 + h * 512 + w] = float32Array[(h * 512 + w) * 3 + c];
        }
      }
    }
    return reshapedArray;
  };

  const reshapeOutput = (flatArray: Float32Array) => {
    const channels = 3;
    const height = 512;
    const width = 512;
    const reshapedArray = new Float32Array(channels * height * width);

    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
          reshapedArray[c * height * width + h * width + w] = flatArray[c * height * width + h * width + w];
        }
      }
    }

    return reshapedArray;
  };

  const transformOutput = (output: Float32Array) => {
    const channels = 3;
    const height = 512;
    const width = 512;
    const transformedArray = new Float32Array(height * width * channels);

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        for (let c = 0; c < channels; c++) {
          transformedArray[h * width * channels + w * channels + c] =
            (output[c * height * width + h * width + w] + 1.0) / 2.0;
        }
      }
    }

    return transformedArray;
  };

  const postprocessOutput = async (output: Float32Array) => {
    const width = 512;
    const height = 512;
    const uint8Array = new Uint8Array(output.length);

    for (let i = 0; i < output.length; i++) {
      uint8Array[i] = Math.min(255, Math.max(0, output[i] * 255));
    }

    const frameData = {
      data: new Uint8Array(width * height * 4),
      width: width,
      height: height,
    };

    for (let i = 0, j = 0; i < uint8Array.length; i += 3, j += 4) {
      frameData.data[j] = uint8Array[i];       // R
      frameData.data[j + 1] = uint8Array[i + 1]; // G
      frameData.data[j + 2] = uint8Array[i + 2]; // B
      frameData.data[j + 3] = 255;             // A (fully opaque)
    }

    const jpegImageData = jpeg.encode(frameData, 90);

    const outputPath = `${RNFS.DocumentDirectoryPath}/output.jpg`;
    await RNFS.writeFile(outputPath, jpegImageData.data.toString('base64'), 'base64');

    return `file://${outputPath}`;
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        <View style={styles.container}>
          <Text style={styles.title}>iFaceAging</Text>
          <View style={styles.separator} lightColor="#eee" darkColor="rgba(255,255,255,0.1)" />

          <TouchableOpacity style={styles.button} onPress={takePicture}>
            <Text style={styles.buttonText}>Take Picture</Text>
          </TouchableOpacity>

          {loading && <ActivityIndicator size="large" color="#007bff" style={styles.loading} />}

          {photoUri && (
            <Image source={{ uri: photoUri }} style={styles.photo} />
          )}

          {resizedPhotoUri && (
            <Image source={{ uri: resizedPhotoUri }} style={styles.photo} />
          )}

          {outputUri && (
            <Image source={{ uri: outputUri }} style={styles.photo} />
          )}

          <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.inputContainer} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#f0f0f0',
  },
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 20,
  },
  container: {
    flex: 1,
    alignItems: 'center',
    backgroundColor: '#ffffff',
    padding: 20,
    borderRadius: 10,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 20,
    textAlign: 'center',
  },
  separator: {
    marginVertical: 10,
    height: 1,
    width: '80%',
    backgroundColor: '#ccc',
    alignSelf: 'center',
  },
  button: {
    backgroundColor: '#007bff',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    alignSelf: 'center',
    marginVertical: 20,
    shadowColor: '#007bff',
    shadowOffset: {
      width: 0,
      height: 10,
    },
    shadowOpacity: 0.3,
    shadowRadius: 13.16,
    elevation: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
  loading: {
    marginVertical: 20,
  },
  photo: {
    width: 300,
    height: 300,
    alignSelf: 'center',
    marginVertical: 20,
    borderRadius: 10,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderTopWidth: 1,
    borderTopColor: '#ddd',
    backgroundColor: '#fff',
  },
});

