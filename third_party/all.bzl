load("//third_party:repos/eigen.bzl", "eigen_repository")
load("//third_party:repos/verum.bzl", "verum_repository")

def dependencies(excludes = []):
    ignores = native.existing_rules().keys() + excludes

    if "com_github_eigenteam_eigen" not in ignores:
        eigen_repository()

def test_dependencies(excludes = []):
    ignores = native.existing_rules().keys() + excludes

    if "com_github_mingkaic_verum" not in ignores:
        verum_repository()

