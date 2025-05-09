import UIKit
import AVFoundation
import Vision
import CoreML

class SquareCameraViewController: UIViewController {
    
    // MARK: - Properties
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    private var currentSampleBuffer: CMSampleBuffer?
    
    // YOLO model
    private var yoloModel: VNCoreMLModel?
    private var detectionOverlay = CALayer()
    
    // YOLO model class labels
    private let classLabels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    // Minimum confidence threshold for detections
    private let confidenceThreshold: Float = 0.5
    
    // Store camera and display orientations
    private var currentDeviceOrientation: UIDeviceOrientation = .portrait
    
    private let previewView: UIView = {
        let view = UIView()
        view.backgroundColor = .black
        view.translatesAutoresizingMaskIntoConstraints = false
        view.contentMode = .scaleAspectFill
        view.clipsToBounds = true
        return view
    }()
    
    private let captureButton: UIButton = {
        let button = UIButton(type: .system)
        button.translatesAutoresizingMaskIntoConstraints = false
        button.setTitle("Take Photo", for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: 18, weight: .medium)
        button.backgroundColor = .white
        button.setTitleColor(.black, for: .normal)
        button.layer.cornerRadius = 25
        button.layer.borderWidth = 2
        button.layer.borderColor = UIColor.black.cgColor
        return button
    }()
    
    private let resultLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "No detections"
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 14)
        return label
    }()
    
    // MARK: - Lifecycle Methods
    override func viewDidLoad() {
        super.viewDidLoad()
        print("viewDidLoad called")
        setupUI()
        setupYOLOModel()
        setupCamera()
        
        // Setup device orientation notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(deviceOrientationDidChange),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        
        // Add button action
        captureButton.addTarget(self, action: #selector(captureButtonTapped), for: .touchUpInside)
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        startSession()
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        stopSession()
    }
    
    // MARK: - Setup Methods
    private func setupUI() {
        print("Setting up UI")
        view.backgroundColor = .black
        
        // Add the preview view
        view.addSubview(previewView)
        view.addSubview(captureButton)
        view.addSubview(resultLabel)
        
        // Make preview view square and centered
        NSLayoutConstraint.activate([
            previewView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            previewView.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            previewView.widthAnchor.constraint(equalTo: view.widthAnchor),
            previewView.heightAnchor.constraint(equalTo: previewView.widthAnchor), // Makes it square
            
            captureButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            captureButton.bottomAnchor.constraint(equalTo: resultLabel.topAnchor, constant: -20),
            captureButton.widthAnchor.constraint(equalToConstant: 150),
            captureButton.heightAnchor.constraint(equalToConstant: 50),
            
            resultLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            resultLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            resultLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            resultLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 40)
        ])
        
        print("UI setup complete")
        
        // Setup detection overlay for drawing bounding boxes
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.frame = previewView.bounds
        detectionOverlay.masksToBounds = true
    }
    
    private func setupYOLOModel() {
        print("Setting up YOLOv8n model")
        do {
            // Load YOLOv8n model
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            // Use the auto-generated model class
            let yolov8nModel = try yolov8n(configuration: config)
            yoloModel = try VNCoreMLModel(for: yolov8nModel.model)
            
            print("YOLOv8n model initialized successfully")
        } catch {
            print("Error setting up YOLO model: \(error)")
        }
    }
    
    private func setupCamera() {
        print("Setting up camera")
        // Configure capture session
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            print("Starting camera configuration")
            self.captureSession.beginConfiguration()
            
            // Set session preset
            if self.captureSession.canSetSessionPreset(.photo) {
                self.captureSession.sessionPreset = .photo
            }
            
            // Setup camera input
            guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: backCamera) else {
                print("Failed to get camera input")
                self.captureSession.commitConfiguration()
                return
            }
            
            print("Got camera input")
            
            // Remove any existing inputs
            self.captureSession.inputs.forEach { self.captureSession.removeInput($0) }
            
            if self.captureSession.canAddInput(input) {
                self.captureSession.addInput(input)
                print("Added camera input")
            }
            
            // Setup video output
            self.videoOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as String): Int(kCVPixelFormatType_32BGRA)]
            self.videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            
            // Remove any existing outputs
            self.captureSession.outputs.forEach { self.captureSession.removeOutput($0) }
            
            if self.captureSession.canAddOutput(self.videoOutput) {
                self.captureSession.addOutput(self.videoOutput)
                print("Added video output")
            }
            
            // Set initial video orientation
            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                    print("Set video orientation")
                }
            }
            
            self.captureSession.commitConfiguration()
            print("Camera configuration committed")
            
            // Setup preview layer on main thread
            DispatchQueue.main.async {
                self.setupPreviewLayer()
            }
        }
    }
    
    private func setupPreviewLayer() {
        print("Setting up preview layer")
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = previewView.bounds
        
        // Remove any existing preview layers
        previewView.layer.sublayers?.forEach { $0.removeFromSuperlayer() }
        
        // Add the preview layer first
        previewView.layer.insertSublayer(previewLayer, at: 0)
        self.previewLayer = previewLayer
        
        // Add the detection overlay on top
        detectionOverlay.frame = previewView.bounds
        previewView.layer.addSublayer(detectionOverlay)
        print("Preview layer setup complete")
    }
    
    // MARK: - Session Control
    private func startSession() {
        print("Starting capture session")
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Check if session is already running
            if self.captureSession.isRunning {
                print("Session already running")
                return
            }
            
            // Check camera authorization status
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                // Start the session
                self.captureSession.startRunning()
                print("Capture session started")
            case .notDetermined:
                // Request authorization
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    if granted {
                        self.captureSession.startRunning()
                        print("Capture session started after authorization")
                    } else {
                        print("Camera access denied")
                    }
                }
            case .denied, .restricted:
                print("Camera access denied or restricted")
            @unknown default:
                print("Unknown authorization status")
            }
        }
    }
    
    private func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        // Update preview layer frame
        previewLayer?.frame = previewView.bounds
        
        // Update detection overlay frame
        detectionOverlay.frame = previewView.bounds
    }
    
    // MARK: - Orientation Handling
    @objc private func deviceOrientationDidChange() {
        guard let connection = videoOutput.connection(with: .video) else {
            return
        }
        
        let deviceOrientation = UIDevice.current.orientation
        guard deviceOrientation.isPortrait || deviceOrientation.isLandscape else {
            return
        }
        
        currentDeviceOrientation = deviceOrientation
        
        // Update video orientation based on device orientation
        let videoOrientation: AVCaptureVideoOrientation
        switch deviceOrientation {
        case .portrait:
            videoOrientation = .portrait
        case .portraitUpsideDown:
            videoOrientation = .portraitUpsideDown
        case .landscapeLeft:
            videoOrientation = .landscapeRight
        case .landscapeRight:
            videoOrientation = .landscapeLeft
        default:
            return
        }
        
        if connection.isVideoOrientationSupported {
            connection.videoOrientation = videoOrientation
        }
    }
    
    // MARK: - YOLO Detection Methods
    private func detectObjects(in pixelBuffer: CVPixelBuffer) {
        guard let yoloModel = self.yoloModel else {
            print("YOLO model not available")
            return
        }
        
        // Create a Vision request with YOLO model
        let request = VNCoreMLRequest(model: yoloModel) { [weak self] request, error in
            if let error = error {
                print("YOLO detection error: \(error)")
                return
            }
            self?.processYOLOResults(request: request)
        }
        
        // Configure the request
        request.imageCropAndScaleOption = .scaleFill
        
        // Create a request handler with the pixel buffer
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        
        // Perform the request
        do {
            try imageRequestHandler.perform([request])
        } catch {
            print("Failed to perform YOLO detection: \(error)")
        }
    }
    
    // YOLO detection structure
    struct YOLODetection {
        let classIndex: Int
        let confidence: Float
        let bbox: CGRect  // Values are normalized (0-1)
    }
    
    private func processYOLOResults(request: VNRequest) {
        // YOLOv8 typically outputs a feature named "output0" - check all available features
        guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else {
            print("No valid observations")
            return
        }
        
        // Get the detection results from the model output
        var detections: [YOLODetection] = []
        
        for observation in observations {
            print("Feature name: \(observation.featureName)")
            
            // YOLOv8 typically has an output named "output0" or similar
            if let multiArray = observation.featureValue.multiArrayValue {
                detections = parseYOLOV8Detections(detection: multiArray)
                break
            }
        }
        
        // Update UI on main thread
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            // Clear previous detections
            self.detectionOverlay.sublayers?.forEach { $0.removeFromSuperlayer() }
            
            // Dictionary to count objects by class
            var objectCounts: [String: Int] = [:]
            
            // Draw new detections
            for detection in detections {
                // Get class name 
                let className = self.classLabels[min(detection.classIndex, self.classLabels.count - 1)]
                
                // Count objects by class
                objectCounts[className, default: 0] += 1
                
                // Draw the detection
                let convertedBox = self.convertYOLOBoundingBox(
                    detection.bbox,
                    toViewWithWidth: self.previewView.bounds.width,
                    height: self.previewView.bounds.height
                )
                
                // Format confidence text
                let confidenceText = String(format: "%.1f%%", detection.confidence * 100)
                
                // Draw the detection
                self.drawDetection(
                    convertedBox,
                    label: "\(className) \(confidenceText)",
                    color: self.colorForClass(className)
                )
            }
            
            // Update result label with counts for each class
            if detections.isEmpty {
                self.resultLabel.text = "No objects detected"
            } else {
                let countStrings = objectCounts.map { "\($0.key): \($0.value)" }
                self.resultLabel.text = countStrings.joined(separator: ", ")
            }
        }
    }
    
    // Parse YOLO v8 model output to get detections
    private func parseYOLOV8Detections(detection: MLMultiArray) -> [YOLODetection] {
        // Create an array to store processed detections
        var detections: [YOLODetection] = []
        
        // Log the shape of the MultiArray to debug
        let shape = detection.shape
        print("Model output shape: \(shape)")
        
        // Different YOLOv8 implementations can have different output formats
        // We'll try to adapt to the actual shape we find
        
        // Get dimensions from the shape
        let dimensions = (0..<shape.count).map { shape[$0].intValue }
        print("Dimensions: \(dimensions)")
        
        // Determine if this is the expected output format
        // YOLO models typically output in one of these formats:
        // - [1, 84, 8400] for COCO (80 classes + 4 bbox)
        // - [1, 8400, 84] transposed version
        // - Single dimension flattened array 
        
        // For simplicity, we'll handle the most common cases
        
        // Try to determine format from dimensions
        if dimensions.count == 3 {
            // Determine if [1, 84, 8400] or [1, 8400, 84] format
            let isFirstFormat = dimensions[1] < dimensions[2]
            
            if isFirstFormat {
                // Format [1, 84, 8400] - columns are detections
                return parseYOLOFormatA(detection: detection, dimensions: dimensions)
            } else {
                // Format [1, 8400, 84] - rows are detections
                return parseYOLOFormatB(detection: detection, dimensions: dimensions)
            }
        }
        
        // Fallback to a basic parser if we can't determine format
        print("Could not determine YOLO output format, using fallback parser")
        return parseYOLOGeneric(detection: detection)
    }
    
    // Parser for [1, 84, 8400] format (classes in first dimension, detections in second)
    private func parseYOLOFormatA(detection: MLMultiArray, dimensions: [Int]) -> [YOLODetection] {
        var detections: [YOLODetection] = []
        
        let numClasses = dimensions[1] - 4 // Subtract 4 for bbox coordinates
        let numDetections = dimensions[2]
        
        // Loop through all potential detections (columns)
        for i in 0..<numDetections {
            // Find the class with highest confidence
            var maxClassIndex = 0
            var maxClassConfidence: Float = 0.0
            
            for j in 0..<numClasses {
                let confidence = detection[[0, 4 + j, i] as [NSNumber]].floatValue
                if confidence > maxClassConfidence {
                    maxClassConfidence = confidence
                    maxClassIndex = j
                }
            }
            
            // Skip low confidence detections
            if maxClassConfidence < confidenceThreshold {
                continue
            }
            
            // Get bounding box coordinates (x_center, y_center, width, height)
            let x = detection[[0, 0, i] as [NSNumber]].floatValue
            let y = detection[[0, 1, i] as [NSNumber]].floatValue
            let width = detection[[0, 2, i] as [NSNumber]].floatValue
            let height = detection[[0, 3, i] as [NSNumber]].floatValue
            
            // Convert from center coordinates to top-left coordinates
            let xMin = x - width/2
            let yMin = y - height/2
            
            // Create normalized bounding box
            let bbox = CGRect(x: CGFloat(xMin), y: CGFloat(yMin), 
                             width: CGFloat(width), height: CGFloat(height))
            
            // Add the detection
            let detection = YOLODetection(
                classIndex: maxClassIndex,
                confidence: maxClassConfidence,
                bbox: bbox
            )
            
            detections.append(detection)
        }
        
        return detections
    }
    
    // Parser for [1, 8400, 84] format (detections in first dimension, classes in second)
    private func parseYOLOFormatB(detection: MLMultiArray, dimensions: [Int]) -> [YOLODetection] {
        var detections: [YOLODetection] = []
        
        let numDetections = dimensions[1]
        let numClasses = dimensions[2] - 4 // Subtract 4 for bbox coordinates
        
        // Loop through all potential detections (rows)
        for i in 0..<numDetections {
            // Find the class with highest confidence
            var maxClassIndex = 0
            var maxClassConfidence: Float = 0.0
            
            for j in 0..<numClasses {
                let confidence = detection[[0, i, 4 + j] as [NSNumber]].floatValue
                if confidence > maxClassConfidence {
                    maxClassConfidence = confidence
                    maxClassIndex = j
                }
            }
            
            // Skip low confidence detections
            if maxClassConfidence < confidenceThreshold {
                continue
            }
            
            // Get bounding box coordinates (x_center, y_center, width, height)
            let x = detection[[0, i, 0] as [NSNumber]].floatValue
            let y = detection[[0, i, 1] as [NSNumber]].floatValue
            let width = detection[[0, i, 2] as [NSNumber]].floatValue
            let height = detection[[0, i, 3] as [NSNumber]].floatValue
            
            // Convert from center coordinates to top-left coordinates
            let xMin = x - width/2
            let yMin = y - height/2
            
            // Create normalized bounding box
            let bbox = CGRect(x: CGFloat(xMin), y: CGFloat(yMin), 
                             width: CGFloat(width), height: CGFloat(height))
            
            // Add the detection
            let detection = YOLODetection(
                classIndex: maxClassIndex,
                confidence: maxClassConfidence,
                bbox: bbox
            )
            
            detections.append(detection)
        }
        
        return detections
    }
    
    // Generic fallback parser
    private func parseYOLOGeneric(detection: MLMultiArray) -> [YOLODetection] {
        // This is a simplified fallback parser that assumes:
        // - First dimension is batch (1)
        // - We'll try to extract at least bbox coordinates and confidence
        
        print("Using generic parser - results may be unreliable")
        var detections: [YOLODetection] = []
        
        // Assuming a generic structure where we might have boxes in some format
        // This is a very crude implementation that might need adjustment
        
        // Try to find potential detections
        let potentialDetections = min(100, detection.count / 5) // Assume at least 4+1 values per detection
        
        for i in 0..<potentialDetections {
            // Try to extract values based on a generic offset - this is very implementation specific
            let baseIndex = i * 5 // Assume at least 5 values per detection (4 box + 1 conf)
            
            guard baseIndex + 4 < detection.count else { break }
            
            // Try to extract confidence - might be at different positions depending on format
            var confidence: Float = 0.0
            if baseIndex + 4 < detection.count {
                confidence = detection[baseIndex + 4].floatValue
            }
            
            // Skip low confidence
            if confidence < confidenceThreshold {
                continue
            }
            
            // Extract coordinates - assuming [x,y,w,h] format
            let x = detection[baseIndex].floatValue
            let y = detection[baseIndex + 1].floatValue
            let width = detection[baseIndex + 2].floatValue
            let height = detection[baseIndex + 3].floatValue
            
            // Convert from center to top-left if needed
            let xMin = x - width/2
            let yMin = y - height/2
            
            // Create bbox
            let bbox = CGRect(x: CGFloat(xMin), y: CGFloat(yMin),
                             width: CGFloat(width), height: CGFloat(height))
            
            // Add detection with a generic class index (0)
            let detection = YOLODetection(
                classIndex: 0,
                confidence: confidence,
                bbox: bbox
            )
            
            detections.append(detection)
        }
        
        return detections
    }
    
    private func colorForClass(_ className: String) -> UIColor {
        // Generate a consistent color for each class name
        let hash = abs(className.hashValue)
        let hue = CGFloat(hash % 256) / 256.0
        return UIColor(hue: hue, saturation: 0.9, brightness: 0.9, alpha: 1.0)
    }
    
    private func convertYOLOBoundingBox(_ boundingBox: CGRect, toViewWithWidth width: CGFloat, height: CGFloat) -> CGRect {
        // Converting from normalized coordinates (0-1) to view coordinates
        let boxWidth = boundingBox.width * width
        let boxHeight = boundingBox.height * height
        
        // x,y in YOLO are normalized coordinates for the top-left corner
        let boxX = boundingBox.minX * width
        let boxY = boundingBox.minY * height
        
        // Create rect in view coordinates
        return CGRect(x: boxX, y: boxY, width: boxWidth, height: boxHeight)
    }
    
    private func drawDetection(_ rect: CGRect, label: String, color: UIColor) {
        // Create box for the detection
        let boxLayer = CALayer()
        boxLayer.frame = rect
        boxLayer.borderWidth = 3.0 // Thicker border for visibility
        boxLayer.borderColor = color.cgColor
        boxLayer.cornerRadius = 4.0
        
        // Create label for the detection
        let textLayer = CATextLayer()
        textLayer.string = label
        textLayer.fontSize = 14
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.backgroundColor = color.withAlphaComponent(0.7).cgColor
        textLayer.cornerRadius = 4.0
        textLayer.masksToBounds = true
        textLayer.alignmentMode = .center
        textLayer.frame = CGRect(x: rect.minX, y: max(0, rect.minY - 20), width: rect.width, height: 20)
        
        // Add to overlay
        detectionOverlay.addSublayer(boxLayer)
        detectionOverlay.addSublayer(textLayer)
    }
    
    // MARK: - Photo Capture
    @objc private func captureButtonTapped() {
        // Get the current video frame
        guard let sampleBuffer = currentSampleBuffer,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get pixel buffer")
            return
        }
        
        // Create a CIImage from the pixel buffer
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Create a CIContext for image processing
        let context = CIContext()
        
        // Convert CIImage to CGImage
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            print("Failed to create CGImage")
            return
        }
        
        // Create a UIImage from CGImage with the correct orientation
        let image = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
        
        // Save the image to photo library
        UIImageWriteToSavedPhotosAlbum(image, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
    }
    
    @objc private func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            print("Error saving photo: \(error.localizedDescription)")
        } else {
            print("Photo saved successfully")
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension SquareCameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Store the current sample buffer
        currentSampleBuffer = sampleBuffer
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        detectObjects(in: pixelBuffer)
    }
} 