import os
import torch
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt

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
        image_paths: list[str],
        image_embedding_size: int,
        show_self_attention: bool,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.image_embedding_size = image_embedding_size
        self.show_self_attention = show_self_attention

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        tensorboard = pl_module.logger.experiment

        for i, image_path in enumerate(self.image_paths):
            # Visualizing root image with the generated caption
            raw_caption, beam_history = pl_module.model.generate_beam_search(image_path, 3)
            generated_caption = pl_module.model.tokenizer.decode(raw_caption)
            image = pl_module.model.image_transform.read_image(image_path).detach().cpu().permute(1, 2, 0)
            captioned_fig = plot_captioned_image(image, generated_caption)
            tensorboard.add_figure(f"{i}_captioned_image", captioned_fig, trainer.current_epoch)
            plt.close(captioned_fig)
            print(generated_caption)

            # Visualizing beam search graph
            graph_fig, nodes = visualize_candidates_graph(beam_history)
            graph_fig.savefig(os.path.join(trainer.log_dir, f"{i}_{trainer.current_epoch}_beam_search_graph.pdf"))
            plt.close(graph_fig)
            node_info = "\n".join(
                [
                    "|".join(
                        [
                            "|" + str(node_id),
                            pl_module.model.tokenizer.decode(node.tokens),
                            str(node.best),
                            str(node.last),
                            "{:.2e}".format(node.probability) + "|",
                        ]
                    )
                    for node_id, node in nodes.items()
                ]
            )
            node_table = f"""
                | *node_id* | *caption* | *in_path* | *best* | *probability* |
                |-----------|-----------|-----------|--------|---------------|
                {node_info}
            """
            table = "\n".join(l.strip() for l in node_table.splitlines())
            tensorboard.add_text(f"{i}_beam_search_labels", table, trainer.current_epoch)

            # Refitting the model to set the attention weights
            dummy_caption = torch.tensor(raw_caption, device=pl_module.device).unsqueeze(0)
            dummy_image = pl_module.model.image_transform.read_image(image_path).unsqueeze(0)
            dummy_image = pl_module.model.image_transform.transform(dummy_image)
            pl_module.model(dummy_image, dummy_caption)

            # Visualizing encoder layers
            dummy_image = pl_module.model.image_transform.denormalize(dummy_image.squeeze(0))
            if self.show_self_attention:
                encoder_heads = extract_encoder_heads(pl_module.model)
                self_att_fig = plot_self_attention(dummy_image, encoder_heads, 3, 4, self.image_embedding_size)
                tensorboard.add_figure(f"{i}_self_attention", self_att_fig, trainer.current_epoch)
                plt.close(self_att_fig)

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
            tensorboard.add_figure(f"{i}_cross_attention", cross_att_fig, trainer.current_epoch)
            plt.close(cross_att_fig)
