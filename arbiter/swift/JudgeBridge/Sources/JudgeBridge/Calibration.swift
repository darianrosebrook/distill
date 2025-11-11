// Sources/JudgeBridge/Calibration.swift
// @author: @darianrosebrook

import Foundation

struct Calibration: Codable {
    struct Platt: Codable {
        let a: Double
        let b: Double
    }
    var score_platt: Platt? = nil
    var clause_temperature: Double? = nil
    var clause_thresholds: [Double]? = nil

    static func load(from url: URL?) -> Calibration {
        guard let url else { return Calibration() }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(Calibration.self, from: data)
        } catch {
            print("[Calibration] Failed to load: \(error)")
            return Calibration()
        }
    }

    func applyScore(_ raw: Double) -> Double {
        if let p = score_platt {
            return 1.0 / (1.0 + exp(-(p.a * raw + p.b)))
        }
        return raw
    }

    func applyClauseLogits(_ logits: [Double]) -> [Double] {
        let t = clause_temperature ?? 1.0
        if t == 1.0 {
            return logits.map { 1.0 / (1.0 + exp(-$0)) }
        }
        return logits.map { 1.0 / (1.0 + exp(-$0 / t)) }
    }
}

