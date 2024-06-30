import React, { useState } from 'react';
import { StyleSheet, TouchableOpacity, Image, KeyboardAvoidingView, Platform, SafeAreaView, ScrollView } from 'react-native';
import { Text, View } from '@/components/Themed';
import { useTensorflowModel } from 'react-native-fast-tflite';
import ImageResizer from 'react-native-image-resizer';
import RNFS from 'react-native-fs';
import { convertToRGB } from 'react-native-image-to-rgb';
import { Buffer } from 'buffer';
import jpeg from 'jpeg-js';

// Ensure Buffer is available globally
global.Buffer = Buffer;

const localImage =     {
  props: {
    source: require("../../assets/images/linkedinphoto.jpeg")
  }
}

export default function TabOneScreen() {
  const objectDetection = useTensorflowModel(require('../../assets/models/aging_gan_generator.tflite'));
  const model = objectDetection.state === 'loaded' ? objectDetection.model : undefined;
  const [resizedPhotoUri, setResizedPhotoUri] = useState<string | null>(null);
  const [outputUri, setOutputUri] = useState<string | null>(null);

  // Function to determine the dimensions of a multi-dimensional array
function getArrayDimensions(arr: any) {
  if (!Array.isArray(arr)) return [];
  const dimensions = [];

  while (Array.isArray(arr)) {
    dimensions.push(arr.length);
    arr = arr[0];
  }

  return dimensions;
}

  const processImage = async () => {
    if (model) {
      // Resize and preprocess the image
      const asset = Image.resolveAssetSource(localImage.props.source);
      const uri = asset.uri;
      console.log(localImage)
      const resizedUri = await resizeImage(uri, 512, 512);
      const inputTensor = await preprocessImage(resizedUri);
      console.log('Input Tensor Dimensions:', getArrayDimensions([inputTensor]));
      const outputDict = model.runSync([inputTensor]);

      // Access the output using the specific key 'output'
      const outputTensor = outputDict[0];
      const reshapedOutput = reshapeOutput(new Float32Array(outputTensor));

      // Transform the output
      const transformedOutput = transformOutput(reshapedOutput);

      // Postprocess the model output to create an image
      const outputUri = await postprocessOutput(transformedOutput);
      setOutputUri(outputUri);
    }
  };

  const resizeImage = async (uri: string, width: number, height: number) => {
    const resizedImage = await ImageResizer.createResizedImage(uri, width, height, 'PNG', 100);
    setResizedPhotoUri(resizedImage.uri);
    return resizedImage.uri;
  };

  const preprocessImage = async (uri: string) => {
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
    // Convert the normalized output to uint8 array
    const uint8Array = new Uint8Array(output.length);
    for (let i = 0; i < output.length; i++) {
      uint8Array[i] = Math.min(255, Math.max(0, output[i] * 255));
    }

    // Convert to RGB format expected by jpeg-js
    const width = 512;
    const height = 512;
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

    const jpegImageData = jpeg.encode(frameData, 90); // 90 is the quality of the output image

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

          <TouchableOpacity style={styles.button} onPress={processImage}>
            <Text style={styles.buttonText}>Process Predefined Image</Text>
          </TouchableOpacity>

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
    backgroundColor: '#f5f5f5',
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
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 20,
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
    paddingHorizontal: 20,
    borderRadius: 8,
    alignSelf: 'center',
    marginVertical: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  photo: {
    width: 200,
    height: 200,
    alignSelf: 'center',
    marginVertical: 20,
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
