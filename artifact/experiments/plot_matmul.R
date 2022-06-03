library(ggplot2)
library(cowplot)
library(sitools)

W=4.804
H=2
S=1
point_size=0.3
line_size=1
linecolors=scale_color_brewer(palette="Set1")
theme = theme_cowplot(font_size=7)

sisec=function(t) {
    t[is.na(t)] = 0
    sitools::f2si(t, 's')
}

args = commandArgs(trailingOnly=TRUE)

{
    n = strtoi(args[1])*strtoi(args[2])*strtoi(args[3])
    tc=function(t)t/n

    data = read.csv(paste0('matmul_custom.csv'), header=T, sep=',')

    data = aggregate(data[, 4:6], list(s0=data$s0, s1=data$s1, s2=data$s2), mean)

    data[, "s2"] = paste0("Out matrix layout: ", data[, "s2"])
    
    ggsave(paste0('plots/heatmap_custom', '_', args[1], '_', args[2], '_', args[3], '.pdf'), device='pdf', units="in", scale=S, width=W, height=H*1.5,
        ggplot(data, aes(x=s1, y=s0, fill=tc(action))) +
        geom_tile() +
        geom_text(aes(label=format(round((tc(action))*1e15, 0), nsmall = 0)), size=1.6) +
        labs(x="Right-hand side matrix layout", y="Left-hand side matrix layout") +
        facet_wrap(~s2, ncol=3) + 
        scale_fill_gradientn(colors=colorspace::diverging_hcl(100, "Blue-Red", l1=66.6, l2=96), name="Normalized time", labels = sisec) +
        theme + background_grid() + theme(legend.position="right") 
    )
}
