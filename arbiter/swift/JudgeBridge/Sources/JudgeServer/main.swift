// Sources/JudgeServer/main.swift
// SwiftNIO HTTP server for CoreML Judge
// @author: @darianrosebrook

import Foundation
import NIO
import NIOHTTP1
import JudgeBridge

struct CompareRequest: Codable {
    let input_ids_a: [Int32]
    let attention_mask_a: [Int32]
    let token_type_ids_a: [Int32]?
    let input_ids_b: [Int32]
    let attention_mask_b: [Int32]
    let token_type_ids_b: [Int32]?
}

struct ScoreResponse: Codable {
    let score: Double
    let clause_probs: [Double]
}

struct CompareResponse: Codable {
    let verdict: String
    let A: ScoreResponse
    let B: ScoreResponse
}

final class Server {
    let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
    let judge: CoreMLJudgeBridge

    init(judge: CoreMLJudgeBridge) {
        self.judge = judge
    }

    func run(host: String = "127.0.0.1", port: Int = 8088) throws {
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.configureHTTPServerPipeline(withErrorHandling: true).flatMap {
                    channel.pipeline.addHandler(HTTPHandler(self.judge))
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
        let channel = try bootstrap.bind(host: host, port: port).wait()
        print("JudgeServer listening on http://\(host):\(port)")
        try channel.closeFuture.wait()
    }
}

final class HTTPHandler: ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart
    let judge: CoreMLJudgeBridge
    var buffer: ByteBuffer?

    init(_ judge: CoreMLJudgeBridge) {
        self.judge = judge
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let part = self.unwrapInboundIn(data)
        switch part {
        case .head(let req):
            buffer = context.channel.allocator.buffer(capacity: 0)
            if req.method == .GET && req.uri == "/health" {
                var headers = HTTPHeaders()
                headers.add(name: "content-type", value: "application/json")
                let body = try! JSONEncoder().encode(["ok": true])
                var buf = context.channel.allocator.buffer(capacity: body.count)
                buf.writeBytes(body)
                let resHead = HTTPResponseHead(version: req.version, status: .ok, headers: headers)
                context.write(self.wrapOutboundOut(.head(resHead)), promise: nil)
                context.write(self.wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
                context.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
            }
        case .body(var chunk):
            if buffer == nil {
                buffer = chunk
            } else {
                buffer!.writeBuffer(&chunk)
            }
        case .end:
            guard var buf = buffer else { return }
            let data = buf.readData(length: buf.readableBytes) ?? Data()
            handleJSON(context: context, data: data)
            buffer = nil
        }
    }

    func handleJSON(context: ChannelHandlerContext, data: Data) {
        let res: (HTTPResponseStatus, Data)
        do {
            if let req = try? JSONDecoder().decode(CompareRequest.self, from: data) {
                let (sa, pa) = try judge.encodeOnce(
                    inputIds: req.input_ids_a,
                    attentionMask: req.attention_mask_a,
                    tokenTypeIds: req.token_type_ids_a ?? nil,
                    length: req.input_ids_a.count
                )
                let (sb, pb) = try judge.encodeOnce(
                    inputIds: req.input_ids_b,
                    attentionMask: req.attention_mask_b,
                    tokenTypeIds: req.token_type_ids_b ?? nil,
                    length: req.input_ids_b.count
                )
                let verdict = sa > sb ? "A" : (sb > sa ? "B" : "TIE")
                let body = CompareResponse(
                    verdict: verdict,
                    A: .init(score: sa, clause_probs: pa),
                    B: .init(score: sb, clause_probs: pb)
                )
                res = (.ok, try JSONEncoder().encode(body))
            } else {
                // /score endpoint expects a single candidate (A fields only)
                struct ScoreRequest: Codable {
                    let input_ids_a: [Int32]
                    let attention_mask_a: [Int32]
                    let token_type_ids_a: [Int32]?
                }
                let sreq = try JSONDecoder().decode(ScoreRequest.self, from: data)
                let (s, p) = try judge.encodeOnce(
                    inputIds: sreq.input_ids_a,
                    attentionMask: sreq.attention_mask_a,
                    tokenTypeIds: sreq.token_type_ids_a ?? nil,
                    length: sreq.input_ids_a.count
                )
                res = (.ok, try JSONEncoder().encode(ScoreResponse(score: s, clause_probs: p)))
            }
        } catch {
            let err = ["error": String(describing: error)]
            res = (.badRequest, try! JSONEncoder().encode(err))
        }
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "application/json")
        let head = HTTPResponseHead(version: .init(major: 1, minor: 1), status: res.0, headers: headers)
        context.write(self.wrapOutboundOut(.head(head)), promise: nil)
        var buf = context.channel.allocator.buffer(capacity: res.1.count)
        buf.writeBytes(res.1)
        context.write(self.wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        context.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
    }
}

// Entry point
if CommandLine.arguments.contains("--help") || CommandLine.arguments.contains("-h") {
    print("Usage: JudgeServer --model <path.mlpackage> --seq-len <N> --clauses <C> [--calibration <calibration.json>] [--host 127.0.0.1] [--port 8088]")
    exit(0)
}

func arg(_ name: String, default d: String? = nil) -> String? {
    if let i = CommandLine.arguments.firstIndex(of: name), i + 1 < CommandLine.arguments.count {
        return CommandLine.arguments[i + 1]
    }
    return d
}

let modelPath = arg("--model")!
let seqLen = Int(arg("--seq-len")!)!
let clauses = Int(arg("--clauses")!)!
let calibPath = arg("--calibration")
let host = arg("--host", default: "127.0.0.1")!
let port = Int(arg("--port", default: "8088")!)!

let judge = try CoreMLJudgeBridge(
    modelURL: URL(fileURLWithPath: modelPath),
    seqLen: seqLen,
    nClauses: clauses,
    calibrationURL: calibPath == nil ? nil : URL(fileURLWithPath: calibPath!)
)
let server = Server(judge: judge)
try server.run(host: host, port: port)

