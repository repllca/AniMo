from animo_io.embedding.hash_embedder import HashTextEmbedder
from animo_io.skeletons.toy_dog import ToySkeletonProvider
from animo_io.generators.animo_lite_generator import AniMoLiteGenerator

def main():
    embedder = HashTextEmbedder(dim=256)
    skel_provider = ToySkeletonProvider()

    gen = AniMoLiteGenerator(
        embedder=embedder,
        skeletons=skel_provider,
        text_dim=256,
        device="cpu",          # GPUがあるなら "cuda"
        weights_path=None,     # 学習したらパスを入れる
        F_feet=2
    )

    out = gen.generate("犬が歩く / a dog is walking", species="dog", T=120)

    print("J =", out.skeleton.J)
    print("T =", out.motion.T)
    print("root_translation:", out.motion.root_translation.shape)
    print("joint_quat:", out.motion.joint_quat.shape)
    print("foot_contacts:", out.motion.foot_contacts.shape)

if __name__ == "__main__":
    main()
