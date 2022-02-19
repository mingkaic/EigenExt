load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def verum_repository():
    git_repository(
        name = "com_github_mingkaic_verum",
        remote = "https://github.com/mingkaic/verum",
        commit = "92350583a5ea8519beeb94a5d73f02b18190ac59",
    )
