#!python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''

scale graphs so ther are the same
add marker for where i stopped gathering last night
add sliding graph and deleting old data?
add percentages to pie chart

'''

plt.style.use('fivethirtyeight')

fig = plt.figure()
grid = plt.GridSpec(2, 3, wspace=0.3, hspace=0.3)
ax1 = fig.add_subplot(grid[1,0])    # rep ups and downs
ax2 = fig.add_subplot(grid[0,0])    # dem ups and downs
ax3 = fig.add_subplot(grid[0,1:])    # both party sentiment
ax4 = fig.add_subplot(grid[1,1])


def animate(i):
    data = open('shutdown_sentiment_data.txt','r').read()
    lines = data.split('\n')
    rtimes = []
    dtimes = []
    times = []

    r_pos_sents = []
    r_neg_sents = []
    d_pos_sents = []
    d_neg_sents = []

    rtime = 0
    dtime = 0
    time = 0

    rep_pos_votes = 0
    rep_neg_votes = 0

    dem_pos_votes = 0
    dem_neg_votes = 0

    d_pos_sent = 0.5
    d_neg_sent = 0.5
    r_pos_sent = 0.5
    r_neg_sent = 0.5

    for line in lines:  # restructure so there is overall time
                        # add in repeat values if its not the party affected
        if 'r' in line:
            rtime += 1
            time += 1
            if 'pos' in line:
                rep_pos_votes += 1
            elif 'neg' in line:
                rep_neg_votes += 1
            
            r_pos_sent = rep_pos_votes / rtime
            r_neg_sent = rep_neg_votes / rtime
            times.append(time)
            r_pos_sents.append(r_pos_sent)
            r_neg_sents.append(r_neg_sent)
            d_pos_sents.append(d_pos_sent)
            d_neg_sents.append(d_neg_sent)
            
        elif 'd' in line:
            dtime += 1
            time += 1
            if 'pos' in line:
                dem_pos_votes += 1
            elif 'neg' in line:
                dem_neg_votes += 1
            
            d_pos_sent = dem_pos_votes / dtime
            d_neg_sent = dem_neg_votes / dtime
            times.append(time)
            d_pos_sents.append(d_pos_sent)
            d_neg_sents.append(d_neg_sent)
            r_pos_sents.append(r_pos_sent)
            r_neg_sents.append(r_neg_sent)
        else:
            time += 1
            times.append(time)
            r_pos_sents.append(r_pos_sent)
            r_neg_sents.append(r_neg_sent)
            d_pos_sents.append(d_pos_sent)
            d_neg_sents.append(d_neg_sent)

    dem_sent = [(x + y)/2 for x, y in zip(d_pos_sents, r_neg_sents)]
    rep_sent = [(x + y)/2 for x, y in zip(r_pos_sents, d_neg_sents)]
    
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    ax1.set_title('Republican Sentiment', fontsize=10)
    ax2.set_title('Democrat Sentiment', fontsize=10)
    ax3.set_title('Party Sentiment', fontsize=10)
    ax4.set_title('Sentiment Split', fontsize=10)

    explode=(0.05, 0.05)
    x_min = 50
    x_max = time
    y_min = -0.1
    y_max = 1.1

    # print past 50 so initial values have time to get replaced
    # use deepskyblue sp that text and other colors stand out better?
    ax1.plot(times[50:], r_pos_sents[50:], label='Republican-Positive',
             color='r')
    ax1.plot(times[50:], r_neg_sents[50:], color='k')
    
    ax2.plot(times[50:], d_pos_sents[50:], label='Democrat-Positive',
             color = 'deepskyblue')
    ax2.plot(times[50:], d_neg_sents[50:], label='Negative-Tweets', color='k')

    ax3.plot(times[50:], dem_sent[50:], color='deepskyblue')
    ax3.plot(times[50:], rep_sent[50:], color='r')

    ax4.pie([dem_sent[-1:], rep_sent[-1:]],
            colors=['deepskyblue','r'], shadow=True, autopct='%1.1f%%',
            explode=explode)

    ax1.axis([x_min, x_max, y_min, y_max])
    ax2.axis([x_min, x_max, y_min, y_max])

    # find a better way to deal with this text
    ax3.text(0, -0.15, 'When both partes are shown on a graph together,',
             fontsize=8)
    ax3.text(0, -0.17, 'the parties are represented by the sum of',
             fontsize=8)
    ax3.text(0, -0.19, 'the positive sentiment for their party and negative'
             , fontsize=8)
    ax3.text(0, -0.21, 'sentiments of the other party', fontsize=8)

    fig.legend(loc=10, bbox_to_anchor=(0.8, 0.2))

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
