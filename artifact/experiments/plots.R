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

# stencil.pdf
{
    data = read.csv('stencil.csv', header=T, sep=',')
    data["compiler"] = 'gcc'
    data["iteration"] = ((seq.int(nrow(data)) - 1) %% 100) + 1

    ggsave("plots/Fig3-stencil.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=iteration,y=time, color=alg)) +
        geom_point(size=point_size) +
        geom_line(stat="smooth", size=line_size, se=FALSE, alpha=0.5) +
        xlab("Iteration")+
        ylab("Time")+
        scale_color_brewer("Indexing", palette="Set1")+
        scale_y_continuous(labels = sisec) +
        facet_wrap(~compiler, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom")
    )
}

# stencil_loop.pdf
{
    data = c()
    
    data = read.csv('stencil.csv', header=T, sep=',')
    data["technique"] = "Indexing"
    
    loop_data = read.csv('stencil_loop.csv', header=T, sep=',')
    loop_data["technique"] = "Indexing+Looping"

    
    data = rbind(data, loop_data)
    data = subset(data, alg == "static")
    data["compiler"] = "gcc"
    data["iteration"] = ((seq.int(nrow(data)) - 1) %% 100) + 1

    ggsave("plots/Fig4-stencil_loop.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=iteration,y=time, color=technique)) +
        geom_point(size=point_size) +
        geom_line(stat="smooth", size=line_size, se=FALSE, alpha=0.5) +
        xlab("Iteration")+
        ylab("Time")+
        scale_color_brewer("Optimization", palette="Set2")+
        scale_y_continuous(labels = sisec) +
        facet_wrap(~compiler, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom")
    )
}

# heatmap_big.pdf
if (length(args) != 0 && args[1] == "ENABLE_CUDA")
{
    n = 10080*10080*10080
    tc=function(t)t/n

    data = read.csv('matmul_big.csv', header=T, sep=',')

    data = aggregate(data[, 4:6], list(s0=data$s0, s1=data$s1, s2=data$s2), mean)

    data[, "s2"] = paste0("Out matrix layout: ", data[, "s2"])
    
    ggsave("plots/FigX-heatmap_all.pdf", device='pdf', units="in", scale=S, width=W, height=H*1.5,
        ggplot(data, aes(x=s1, y=s0, fill=tc(action))) +
        geom_tile() +
        geom_text(aes(label=format(round((tc(action))*1e15, 0), nsmall = 0)), size=1.6) +
        labs(x="Right-hand side matrix layout", y="Left-hand side matrix layout") +
        facet_wrap(~s2, ncol=3) + 
        scale_fill_gradientn(colors=colorspace::diverging_hcl(100, "Blue-Red", l1=66.6, l2=96), name="Normalized time", labels = sisec) +
        theme + background_grid() + theme(legend.position="right") 
    )
}

# matmul.pdf
if (length(args) != 0 && args[1] == "ENABLE_CUDA")
{
    n = 10080*10080*10080
    tc=function(t)t/n
    
    data = read.csv('matmul_big.csv', header=T, sep=',')

    data = aggregate(data[, 4:6], list(s0=data$s0, s1=data$s1, s2=data$s2), mean)
    
    data = subset(data, (s0 == "RC" & s1 == "RC" & s2 == "R") | (s0 == "RR" & s1 == "RC" & s2 == "R") | (s0 == "R" & s1 == "R" & s2 == "R") | (s0 == "R" & s1 == "C" & s2 == "R") | (s0 == "C" & s1 == "C" & s2 == "R"))

    norm = data[data$s0 == "R" & data$s1 == "R" & data$s2 == "R", ]["action"]

    data["nn"] = 0
    data["diff"] = 0
    data["speedup"] = 0

    norm = as.numeric(norm)
    data <- transform(data, nn = norm)
    data <- transform(data, diff = action - norm)
    data <- transform(data, speedup = 1 - (action / norm))

    ggsave("plots/Fig1-matmul.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=reorder(paste0(paste(s0, s1, sep="x"), "=R"), transform), y=tc(action), fill=paste(s0, s1, sep="x"))) +
        geom_bar(stat = "identity") +
        geom_errorbar(aes(ymin=tc(action), ymax=tc(action - diff)), width=.1) + 
        geom_text(aes(label=ifelse(speedup == 0, "", scales::percent(abs(speedup))), y=tc(nn)), size=1.5, vjust=0.5*ifelse(data$diff < 0, -1, 3)) + 
        ylab("Normalized time") +
        xlab("Layout configuration") +
        geom_hline(aes(yintercept=tc(nn)), size=.3) +
        scale_y_continuous(labels=sisec) +
        scale_fill_brewer(palette="Set3") +
        theme + background_grid() + theme(legend.position="none", axis.text.x = element_text(angle = 45, hjust=1))
    )
}

# matmul_transform.pdf
if (length(args) != 0 && args[1] == "ENABLE_CUDA")
{
    data = read.csv('matmul_big.csv', header=T, sep=',')
    data["example"] = "Output matrix size: 10080x10080"
    
    data_small = read.csv('matmul_small.csv', header=T, sep=',')
    data_small["example"] = "Output matrix size: 1008x1008"

    data = rbind(data, data_small)

    data = aggregate(data[, 4:6], list(s0=data$s0, s1=data$s1, s2=data$s2, example=data$example), mean)

    data2 = data[,1:4]

    data2["time"] = data[, 5]
    data2["type"] = "transform"
    data2["ac"] = data[, 6]

    data3 = data[,1:4]

    data3["time"] = data[, 6]
    data3["type"] = "action"
    data3["ac"] = data[, 6]
    data = rbind(data2, data3)

    data = subset(data, (s0 == "RR" & s1 == "RC" & s2 == "RR") | (s0 == "R" & s1 == "R" & s2 == "R") | (s0 == "R" & s1 == "C" & s2 == "R") | (s0 == "C" & s1 == "C" & s2 == "R"))

    ggsave("plots/Fig2-matmul_transform.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(y=reorder(paste(paste(s0, s1, sep="x"), s2, sep="="), ac), x=time, fill=type)) +
        geom_col() +
        ylab("Layout configuration") +
        xlab("Time") +
        scale_x_continuous(labels=sisec) +
        scale_fill_brewer(name="Time spent in", labels=c("matmul", "transform"), palette="Pastel1", direction=-1) +
        facet_wrap(~example, scales="free_x") +
        theme + background_grid() + theme(legend.position="bottom", axis.text.y = element_text(angle = 45, hjust=1))
    )
}
