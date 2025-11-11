// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "JudgeBridge",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "JudgeBridge", type: .dynamic, targets: ["JudgeBridge"]),
        .executable(name: "JudgeServer", targets: ["JudgeServer"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.60.0")
    ],
    targets: [
        .target(
            name: "JudgeBridge",
            dependencies: [],
            path: "Sources/JudgeBridge",
            resources: [.process("Resources")],
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "JudgeServer",
            dependencies: [
                .product(name: "NIO", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                "JudgeBridge"
            ],
            path: "Sources/JudgeServer"
        )
    ]
)

