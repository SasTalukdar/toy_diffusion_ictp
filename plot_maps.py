def map_plot(x,y,fld,title,day,ifig,exp,vmin,vmax,loc):
    fig,ax=plt.subplots()
    print (vmin,vmax)
    img=ax.pcolormesh(x,y,fld,cmap='Spectral',vmin=vmin,vmax=vmax)
    ax.set_title(title+" : day "+day)
    #ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xlabel('km', fontsize=15)
    ax.set_ylabel('km', fontsize=15)
    ax.set_xlim(0,int(domain_x/1000.))
    ax.set_ylim(0,int(domain_y/1000.))
    # fudge on resolution
    ax.scatter(dx/1000*loc[:,1],dy/1000*loc[:,0],s=10,marker="s",zorder=1,color="black",edgecolors="white")
    plt.colorbar(img,ax=ax)
    plt.savefig(odir+"map_"+title+"_"+exp+"_"+str(ifig).zfill(3)+".png")
    plt.close(fig)



    # other plot stuff

    
    if lplot:
        fig1,ax1=plt.subplots()
        fig_ts,ax_ts=plt.subplots(nrows=3,ncols=2,sharex=True)

    axc=np.unravel_index(range(6),(2,3))


            if lplot:
            ax_ts[0,0].plot(days,ncnv,label=sdiurn)



                if lplot and (it*dt)%(nfig_hr*3600)==0 and day>sfig_day:

                    print("PLOT: day ",sday)        
                    map_plot(x,y,cin[1,:,:],"CIN",sday,ifig,sdiurn,0,1,cnv_loc)
                    map_plot(x,y,crh[1,:,:],"CRH",sday,ifig,sdiurn,0,1,cnv_loc)
                    map_plot(x,y,prob,"P",sday,ifig,sdiurn,0,np.max(prob),cnv_loc)
                    map_plot(x,y,prob_cin,"P-CIN",sday,ifig,sdiurn,0,np.max(prob_cin),cnv_loc)
                    map_plot(x,y,prob_crh,"P-CRH",sday,ifig,sdiurn,0,np.max(prob_crh),cnv_loc)

                    # correct these surface plots
                    if False:
                        fig=plt.figure()
                        ax=plt.axes(projection='3d')
                        ax.plot_surface(x,y,crh[1,:,:],cmap='viridis', edgecolor='none')
                        ax.set_title('CRH: day '+str(ifig*nfig_hr/24))
                        ax.set_zlim([0.4,1])
                        fig.savefig("crh_3d_"+str(ifig).zfill(3)+".png")
                        plt.close(fig)

                        fig=plt.figure()
                        ax=plt.axes(projection='3d')
                        ax.plot_surface(x,y,cin[1,:,:],cmap='viridis', edgecolor='none')
                        ax.set_title('CIN p: day '+str(ifig*nfig_hr/24))
                        ax.set_zlim([0,1])
                        fig.savefig("cin_3d_"+str(ifig).zfill(3)+".png")
                        plt.close(fig)

                    # line plot
                    ax1.plot(x[mp,mp-40:mp+40],crh[1,mp,mp-40:mp+40])
                    ax1.plot(x[mp,mp-40:mp+40],cin[1,mp,mp-40:mp+40])

                    ifig+=1


        if lplot:
            ax_ts[0,1].plot(days,crh_mean,label=sdiurn)
            ax_ts[0,1].set_xlabel("Time(days)")
            ax_ts[0,1].set_ylabel("CRH mean ")

            ax_ts[1,0].plot(days,crh_std,label=sdiurn)
            ax_ts[1,0].set_xlabel("Time(days)")
            ax_ts[1,0].set_ylabel("CRH standard dev ")

            ax_ts[1,1].plot(days,crh_in_new,label=sdiurn)
            ax_ts[1,1].set_xlabel("Time(days)")
            ax_ts[1,1].set_ylabel("CRH conv new ")

            ax_ts[2,0].plot(days,crh_driest,label=sdiurn)
            ax_ts[2,0].set_xlabel("Time(days)")
            ax_ts[2,0].set_ylabel("CRH min ")
            ax_ts[0,1].legend()

    # end of run
    if lplot:
        fig_ts.savefig(odir+"timeseries_"+sdiurn+".pdf")
        fig1.savefig(odir+"lineplot.png")
        plt.close(fig_ts)
        plt.close(fig1)
            
