// metro.config.js
const { getDefaultConfig } = require('expo/metro-config');

module.exports = (async () => {
  const defaultConfig = await getDefaultConfig(__dirname);
  const { resolver, transformer } = defaultConfig;

  return {
    ...defaultConfig,
    transformer: {
      ...transformer,
      assetPlugins: ['expo-asset/tools/hashAssetFiles'],
    },
    resolver: {
      ...resolver,
      assetExts: [...resolver.assetExts, 'png', 'jpg', ,'jpeg', 'bin', 'tflite'],
      sourceExts: [...resolver.sourceExts, 'jsx', 'js', 'ts', 'tsx'],
    },
  };
})();