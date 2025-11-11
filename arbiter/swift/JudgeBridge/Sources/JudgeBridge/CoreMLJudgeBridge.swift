// Sources/JudgeBridge/CoreMLJudgeBridge.swift
// @author: @darianrosebrook

import Foundation
import CoreML

public final class CoreMLJudgeBridge {
    public let model: MLModel
    public let seqLen: Int
    public let nClauses: Int
    public let calibration: Calibration

    public init(modelURL: URL, seqLen: Int, nClauses: Int, calibrationURL: URL? = nil, computeUnits: MLComputeUnits = .all) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: modelURL, configuration: cfg)
        self.seqLen = seqLen
        self.nClauses = nClauses
        self.calibration = Calibration.load(from: calibrationURL)
    }

    private func makeI32Array(_ data: UnsafePointer<Int32>, count: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: count)], dataType: .int32)
        // MLMultiArray uses Int32 pointer for int32
        arr.withUnsafeMutableBytes { (buf: UnsafeMutableRawBufferPointer) in
            let dst = buf.bindMemory(to: Int32.self)
            dst.baseAddress!.assign(from: data, count: count)
        }
        return arr
    }

    /// Run single branch (encode_once). Returns (score, clauseProbs)
    public func encodeOnce(inputIds: UnsafePointer<Int32>, attentionMask: UnsafePointer<Int32>, tokenTypeIds: UnsafePointer<Int32>?, length: Int) throws -> (Double, [Double]) {
        precondition(length == seqLen, "Input length \(length) must equal seqLen \(seqLen) for this model")
        let ids = try makeI32Array(inputIds, count: length)
        let mask = try makeI32Array(attentionMask, count: length)
        let tt: MLMultiArray
        if let ttPtr = tokenTypeIds {
            tt = try makeI32Array(ttPtr, count: length)
        } else {
            // allocate zero token types if exporter wired it that way
            tt = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
            memset(tt.dataPointer, 0, length * MemoryLayout<Int32>.size)
        }
        let inputs: [String: Any] = [
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": tt
        ]
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))

        // score is scalar, clause_logits is [C]
        let scoreVal = out.featureValue(for: "score")!
        let rawScore: Double
        if scoreVal.type == .multiArray, let ma = scoreVal.multiArrayValue {
            rawScore = Double(truncating: ma[0])
        } else {
            rawScore = scoreVal.doubleValue
        }

        let logitsMA = out.featureValue(for: "clause_logits")!.multiArrayValue!
        var logits: [Double] = Array(repeating: 0.0, count: nClauses)
        for i in 0..<nClauses {
            logits[i] = Double(truncating: logitsMA[i])
        }

        let calibratedScore = calibration.applyScore(rawScore)
        let clauseProbs = calibration.applyClauseLogits(logits)

        return (calibratedScore, clauseProbs)
    }
}

