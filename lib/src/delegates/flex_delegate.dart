/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import 'dart:ffi';
import 'dart:io';

import 'package:quiver/check.dart';
import 'package:tflite_flutter/src/bindings/tensorflow_lite_bindings_generated.dart';

import '../delegate.dart';

const _flexLibraryName = 'libtensorflowlite_flex_jni.so';
const _jniPrefix = 'Java_org_tensorflow_lite_flex_FlexDelegate_';

// The flex library only exports the JNI entry points of the Java
// `FlexDelegate` class. None of them use the `JNIEnv*` / `jclass` arguments,
// so they are called directly with null for both.
typedef _InitNative = Void Function(Pointer<Void> env, Pointer<Void> clazz);
typedef _InitDart = void Function(Pointer<Void> env, Pointer<Void> clazz);
typedef _CreateNative = Int64 Function(Pointer<Void> env, Pointer<Void> clazz);
typedef _CreateDart = int Function(Pointer<Void> env, Pointer<Void> clazz);
typedef _DeleteNative = Void Function(
    Pointer<Void> env, Pointer<Void> clazz, Int64 delegate);
typedef _DeleteDart = void Function(
    Pointer<Void> env, Pointer<Void> clazz, int delegate);

class _FlexBindings {
  _FlexBindings._(DynamicLibrary lib)
      : create = lib.lookupFunction<_CreateNative, _CreateDart>(
          '${_jniPrefix}nativeCreateDelegate',
        ),
        delete = lib.lookupFunction<_DeleteNative, _DeleteDart>(
          '${_jniPrefix}nativeDeleteDelegate',
        ) {
    lib.lookupFunction<_InitNative, _InitDart>(
      '${_jniPrefix}nativeInitTensorFlow',
    )(nullptr, nullptr);
  }

  final _CreateDart create;
  final _DeleteDart delete;

  /// Loaded on first use so that apps which do not bundle the flex library
  /// are unaffected.
  static final _FlexBindings instance = () {
    if (!Platform.isAndroid) {
      throw UnsupportedError('FlexDelegate is only supported on Android.');
    }
    final DynamicLibrary lib;
    try {
      lib = DynamicLibrary.open(_flexLibraryName);
    } on ArgumentError catch (e) {
      throw UnsupportedError(
        'Could not load $_flexLibraryName. Add the '
        '"org.tensorflow:tensorflow-lite-select-tf-ops" dependency to your '
        "app's android/app/build.gradle to use FlexDelegate. ($e)",
      );
    }
    return _FlexBindings._(lib);
  }();
}

/// Flex delegate for Android.
///
/// Runs select TensorFlow ops that have no TFLite builtin equivalent. See
/// https://ai.google.dev/edge/litert/models/ops_select
///
/// The native library is large and is not bundled with this plugin. To use
/// this delegate, add the dependency to your app's `android/app/build.gradle`:
///
/// ```gradle
/// dependencies {
///     implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1'
/// }
/// ```
///
/// Throws [UnsupportedError] on other platforms or if the library is missing.
class FlexDelegate implements Delegate {
  final Pointer<TfLiteDelegate> _delegate;
  bool _deleted = false;

  @override
  Pointer<TfLiteDelegate> get base => _delegate;

  FlexDelegate._(this._delegate);

  factory FlexDelegate() {
    final address = _FlexBindings.instance.create(nullptr, nullptr);
    if (address == 0) {
      throw StateError('Failed to create FlexDelegate.');
    }
    return FlexDelegate._(Pointer.fromAddress(address));
  }

  @override
  void delete() {
    checkState(!_deleted, message: 'FlexDelegate already deleted.');
    _FlexBindings.instance.delete(nullptr, nullptr, _delegate.address);
    _deleted = true;
  }
}
