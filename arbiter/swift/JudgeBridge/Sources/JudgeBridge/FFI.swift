// Sources/JudgeBridge/FFI.swift
// C FFI helpers for CoreML Judge Bridge
// @author: @darianrosebrook

import Foundation

// C FFI helpers
private func takeOpaque(_ p: UnsafeMutableRawPointer?) -> CoreMLJudgeBridge? {
    guard let p else { return nil }
    return Unmanaged<CoreMLJudgeBridge>.fromOpaque(p).takeUnretainedValue()
}

@_cdecl("judge_create")
public func judge_create(model_path: UnsafePointer<CChar>?, seq_len: Int32, n_clauses: Int32, calibration_path: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    guard let cpath = model_path else { return nil }
    let path = String(cString: cpath)
    let calib = calibration_path != nil ? String(cString: calibration_path!) : nil
    do {
        let bridge = try CoreMLJudgeBridge(
            modelURL: URL(fileURLWithPath: path),
            seqLen: Int(seq_len),
            nClauses: Int(n_clauses),
            calibrationURL: calib == nil ? nil : URL(fileURLWithPath: calib!)
        )
        return Unmanaged.passRetained(bridge).toOpaque()
    } catch {
        fputs("judge_create error: \(error)\n", stderr)
        return nil
    }
}

@_cdecl("judge_destroy")
public func judge_destroy(handle: UnsafeMutableRawPointer?) {
    if let handle {
        Unmanaged<CoreMLJudgeBridge>.fromOpaque(handle).release()
    }
}

@_cdecl("judge_encode_once")
public func judge_encode_once(
    handle: UnsafeMutableRawPointer?,
    input_ids: UnsafePointer<Int32>?,
    attention_mask: UnsafePointer<Int32>?,
    token_type_ids: UnsafePointer<Int32>?,
    length: Int32,
    out_score: UnsafeMutablePointer<Float>?,
    out_clause_probs: UnsafeMutablePointer<Float>?,
    out_clause_len: Int32
) -> Int32 {
    guard let h = takeOpaque(handle),
          let ids = input_ids,
          let mask = attention_mask,
          let outS = out_score,
          let outP = out_clause_probs else {
        return -1
    }
    if out_clause_len != Int32(h.nClauses) {
        return -2
    }
    do {
        let (score, probs) = try h.encodeOnce(
            inputIds: ids,
            attentionMask: mask,
            tokenTypeIds: token_type_ids,
            length: Int(length)
        )
        outS.pointee = Float(score)
        for i in 0..<h.nClauses {
            outP[i] = Float(probs[i])
        }
        return 0
    } catch {
        fputs("judge_encode_once error: \(error)\n", stderr)
        return -3
    }
}

