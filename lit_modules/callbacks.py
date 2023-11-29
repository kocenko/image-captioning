import os
import torch
from lightning.pytorch.callbacks import Callback

from evaluation.extract_heads import aggregate_heads, extract_decoder_heads, extract_encoder_heads
from evaluation.visualizing import (
    plot_self_attention,
    plot_cross_attention,
    plot_captioned_image,
    visualize_candidates_graph,
)


class GenerateCaption(Callback):
    def __init__(
        self,
        image_path: str,
        image_embedding_size: int,
        show_self_attention: bool,
    ):
        super().__init__()
        self.image_path = image_path
        self.image_embedding_size = image_embedding_size
        self.show_self_attention = show_self_attention

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        tensorboard = pl_module.logger.experiment

        # Visualizing root image with the generated caption
        raw_caption, beam_history = pl_module.model.generate_beam_search(self.image_path, 3)
        generated_caption = pl_module.model.tokenizer.decode(raw_caption)
        image = pl_module.model.image_transform.read_image(self.image_path).permute(1, 2, 0)
        captioned_fig = plot_captioned_image(image, generated_caption)
        tensorboard.add_figure("captioned_image", captioned_fig, trainer.current_epoch)
        print(generated_caption)

        # Visualizing beam search graph
        graph_fig, nodes = visualize_candidates_graph(beam_history)
        graph_fig.savefig(os.path.join(trainer.log_dir, f"{trainer.current_epoch}_beam_search_graph.pdf"))
        node_info = "\n".join(
            [
                "|".join(
                    [
                        "|" + str(node_id),
                        pl_module.model.tokenizer.decode(node.tokens),
                        str(node.best),
                        str(node.last) + "|",
                    ]
                )
                for node_id, node in nodes.items()
            ]
        )
        node_table = f"""
            | *node_id* | *caption* | *in_path* | *best* |
            |-----------|-----------|-----------|--------|
            {node_info}
        """
        table = "\n".join(l.strip() for l in node_table.splitlines())
        tensorboard.add_text("beam_search_labels", table, trainer.current_epoch)

        # Refitting the model to set the attention weights
        dummy_caption = torch.tensor(raw_caption, device=pl_module.device).unsqueeze(0)
        dummy_image = pl_module.model.image_transform.read_image(self.image_path).unsqueeze(0)
        dummy_image = pl_module.model.image_transform.transform(dummy_image)
        pl_module.model(dummy_image, dummy_caption)

        # Visualizing encoder layers
        dummy_image = pl_module.model.image_transform.denormalize(dummy_image.squeeze(0))
        if self.show_self_attention:
            encoder_heads = extract_encoder_heads(pl_module.model)
            self_att_fig = plot_self_attention(dummy_image, encoder_heads, 3, 4, self.image_embedding_size)
            tensorboard.add_figure("self_attention", self_att_fig, trainer.current_epoch)

        decoder_heads = extract_decoder_heads(pl_module.model)
        aggregated_heads = aggregate_heads(decoder_heads, method="sum")
        cross_att_fig = plot_cross_attention(
            dummy_image,
            raw_caption,
            aggregated_heads,
            10,
            self.image_embedding_size,
            pl_module.model.tokenizer.decode_map,
        )
        tensorboard.add_figure("cross_attention", cross_att_fig, trainer.current_epoch)
